"""Tests for the five signed-off GAN fixes (analysis/gan_tuning/CODE_FIXES.md).

  FIX-1  ``ladd_gside_checkpoint_recover``       G-side activation-ckpt recovery
  FIX-2  ``ladd_action_origin_track_slice``      action origin tracks the slice
  FIX-3  ``gan_health_show_r1_fire_rate`` /
         ``gan_health_show_hold_step``           R1 telemetry on the health line
  FIX-4  ``ladd_match_distance_log`` /
         ``ladd_gt_transition_match_pool_cap``   match distance + pool width
  FIX-5  ``ladd_backbone_grad_scale_check``      backbone grad-scale audit

Every fix gets, at minimum:
  * a DEFAULT-OFF test proving the pre-change behaviour is reproduced, with
    an explicit MUTATION CONTROL showing the assertion would fail if the
    behaviour had changed; and
  * a positive test of the ON behaviour.

CPU-only, no CUDA, no dataset. ``wan/modules/t5.py`` evaluates
``torch.cuda.current_device()`` at import, so imports are patched exactly
as ``testing/test_r1_positional_deferred.py`` does.

Run:
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
        python -m pytest testing/test_gan_code_fixes.py -q
"""
import inspect
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT
    from model.ladd_disc import LADDRegisterReadout

Trainer = CAFT.ActionForcingDMDTrainer


# ===========================================================================
# Shared fixtures
# ===========================================================================
def make_trainer(**cfg):
    """A bare trainer instance carrying only the attributes the helpers read."""
    t = Trainer.__new__(Trainer)
    t.config = SimpleNamespace(**cfg)
    t.model = SimpleNamespace()
    t.step = 0
    t.is_main_process = True
    return t


def make_readout(use_checkpoint=True):
    """A tiny but REAL ``LADDRegisterReadout`` (2 taps, dim 8)."""
    with patch.object(torch.cuda, "current_device", return_value=0):
        r = LADDRegisterReadout(
            block_indices=[0, 1], dim_teacher=8, blocks_per_token=1,
            block_ffn_dim=16, block_num_heads=2, head_hidden_dim=8,
            head_num_layers=2, head_dropout=0.0, num_class=1,
            use_checkpoint=use_checkpoint,
        )
    # The gen-side guidance window always runs the disc in ``eval()`` (see
    # the spectral-norm note at the call site), so mirror that here or the
    # power iteration mutates ``_sigma`` between the two comparison runs.
    r.eval()
    return r


def count_tap_forwards(readout):
    """Instrument ``_tap_forward``.

    Calls == n_taps            -> the checkpoint did NOT arm (verbatim path,
                                  every intermediate retained until backward)
    Calls == 2 * n_taps        -> the checkpoint ARMED (record + backward-time
                                  replay), i.e. the ~30 GiB is recovered.
    """
    calls = {"n": 0}
    orig = readout._tap_forward

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    readout._tap_forward = spy
    return calls


def make_feats(x0, x1):
    return {0: x0.clone().requires_grad_(True), 1: x1.clone().requires_grad_(True)}


# ===========================================================================
# FIX-1 -- G-side activation-checkpoint recovery
# ===========================================================================
class TestFix1ArmingPredicate:
    """The mechanism the diagnosis names: ``model/ladd_disc.py`` arms the
    per-tap checkpoint only when the readout's OWN params require grad."""

    def test_checkpoint_disarms_when_disc_is_frozen(self):
        """PRE-CHANGE behaviour: the gen-side window froze the disc, so the
        checkpoint never armed and every fp32 readout intermediate was held
        until ``generator_loss.backward()``. This is the ~34 GiB."""
        r = make_readout()
        r.requires_grad_(False)          # what trainer:13338 used to do
        calls = count_tap_forwards(r)
        x = torch.randn(2, 5, 8)
        f = make_feats(x, x)
        r(f).sum().backward()
        assert calls["n"] == 2, (
            "expected ONE _tap_forward per tap (verbatim, un-checkpointed); "
            f"got {calls['n']}")

    def test_checkpoint_arms_when_disc_is_left_trainable(self):
        """POST-FIX behaviour: leaving requires_grad ON makes the predicate
        stable across record and replay, so the checkpoint arms and the
        activations are recomputed at backward time instead of retained."""
        r = make_readout()
        assert all(p.requires_grad for p in r.parameters())
        calls = count_tap_forwards(r)
        x = torch.randn(2, 5, 8)
        f = make_feats(x, x)
        r(f).sum().backward()
        assert calls["n"] == 4, (
            "expected TWO _tap_forward per tap (record + backward replay) = "
            f"the checkpoint armed; got {calls['n']}")

    def test_mutation_control_arming_counts_differ(self):
        """MUTATION CONTROL for the two tests above: if the arming predicate
        stopped depending on the params' requires_grad, both counts would be
        equal and the two assertions could not both hold."""
        x = torch.randn(2, 5, 8)
        counts = []
        for freeze in (True, False):
            r = make_readout()
            if freeze:
                r.requires_grad_(False)
            c = count_tap_forwards(r)
            r(make_feats(x, x)).sum().backward()
            counts.append(c["n"])
        assert counts[0] != counts[1], (
            "arming is supposed to be requires_grad-dependent; identical "
            f"tap-forward counts {counts} means the predicate changed")

    def test_generator_gradient_is_preserved(self):
        """The adversarial gradient the GENERATOR receives (d loss / d feat)
        must not change materially when the disc is left trainable.

        NOT bit-identical: unfreezing the params changes the backward
        kernel-dispatch path, which moves the result by ~1e-7 relative --
        fp noise, three orders of magnitude below anything the GAN weight
        or the ramp does. Asserted at that scale, deliberately."""
        x0 = torch.randn(2, 5, 8)
        x1 = torch.randn(2, 5, 8)
        r = make_readout()

        r.requires_grad_(False)
        f_frozen = make_feats(x0, x1)
        out_frozen = r(f_frozen)
        out_frozen.sum().backward()

        r.requires_grad_(True)
        r.zero_grad(set_to_none=True)
        f_live = make_feats(x0, x1)
        out_live = r(f_live)
        out_live.sum().backward()

        assert torch.allclose(
            out_frozen.detach(), out_live.detach(), atol=1e-6, rtol=1e-5)
        for k in f_frozen:
            assert torch.allclose(
                f_frozen[k].grad, f_live[k].grad, atol=1e-6, rtol=1e-5), (
                f"generator-side gradient for tap {k} moved by more than fp "
                "noise")

    def test_frozen_path_leaks_no_disc_gradient(self):
        """Baseline for the correctness requirement: with the disc frozen,
        the generator backward deposits NOTHING on the disc's params."""
        r = make_readout()
        r.requires_grad_(False)
        x = torch.randn(2, 5, 8)
        r(make_feats(x, x)).sum().backward()
        assert all(p.grad is None for p in r.parameters())

    def test_trainable_path_DOES_leak_and_that_is_what_must_be_discarded(self):
        """The cost of the fix, made explicit: with the disc left trainable
        the generator backward DOES write ``.grad`` on every disc param.

        This is precisely the gradient that would silently train the
        discriminator on the GENERATOR's objective, and it is why
        ``_ladd_gside_release_disc_grads`` exists."""
        r = make_readout()
        x = torch.randn(2, 5, 8)
        r(make_feats(x, x)).sum().backward()
        n = sum(1 for p in r.parameters() if p.grad is not None)
        assert n == len(list(r.parameters())) and n > 0


class TestFix1Discard:
    """``_ladd_gside_release_disc_grads`` -- the correctness half."""

    def _armed_trainer(self, flag):
        t = make_trainer(ladd_gside_checkpoint_recover=flag)
        t.r3gan_disc = make_readout()
        return t

    def test_flag_defaults_off(self):
        t = make_trainer()
        assert t._ladd_gside_ckpt_recover() is False

    def test_default_off_release_is_a_no_op(self):
        """DEFAULT-OFF BYTE-IDENTITY: with the flag off the gen-side sites
        never set the latch, so the helper touches nothing -- not the
        grads, not the log dict."""
        t = self._armed_trainer(False)
        x = torch.randn(2, 5, 8)
        t.r3gan_disc(make_feats(x, x)).sum().backward()
        out = {}
        t._ladd_gside_release_disc_grads(out)
        assert out == {}
        assert any(p.grad is not None for p in t.r3gan_disc.parameters()), (
            "MUTATION CONTROL: the helper must not clear grads when the "
            "latch was never set")

    def test_on_the_disc_receives_no_generator_gradient(self):
        """THE CRITICAL CORRECTNESS TEST.

        Replays the exact edited sequence: gen-side window (no freeze, latch
        set) -> generator backward -> discard. After the discard EVERY disc
        parameter must read ``.grad is None``, so the next D-update's
        ``r3gan_optimizer.step()`` can carry no generator contribution."""
        t = self._armed_trainer(True)
        disc = t.r3gan_disc

        # --- the edited gen-side window ---
        gsr = t._ladd_gside_ckpt_recover()
        assert gsr is True
        if gsr:
            t._ladd_gside_disc_grads_dirty = True
        else:                                   # pragma: no cover
            disc.requires_grad_(False)
        was_training = disc.training
        disc.eval()
        try:
            x = torch.randn(2, 5, 8)
            gen_gan_loss = disc(make_feats(x, x)).sum()
        finally:
            if not gsr:                          # pragma: no cover
                disc.requires_grad_(True)
            if was_training:                     # pragma: no cover
                disc.train()

        gen_gan_loss.backward()
        assert any(p.grad is not None for p in disc.parameters()), (
            "precondition: the generator backward really did reach the "
            "disc's weights (otherwise the test proves nothing)")

        out = {}
        t._ladd_gside_release_disc_grads(out)

        assert all(p.grad is None for p in disc.parameters()), (
            "LEAK: a disc parameter still carries a generator gradient after "
            "the discard -- the disc would be trained on the generator "
            "objective")
        assert out["train/ladd_gside_disc_grad_zeroed"] == 1.0
        assert out["train/ladd_gside_disc_grad_params"] == float(
            len(list(disc.parameters())))
        assert out["train/ladd_gside_disc_grad_norm"] > 0.0, (
            "the DISCARDED norm must be > 0 -- a zero here would mean the "
            "gen-side forward never reached the disc's weights at all")

    def test_mutation_control_skipping_the_discard_leaves_the_leak(self):
        """MUTATION CONTROL for the test above: drop the discard call and the
        leak assertion fails, so that test is not vacuous."""
        t = self._armed_trainer(True)
        disc = t.r3gan_disc
        t._ladd_gside_disc_grads_dirty = True
        x = torch.randn(2, 5, 8)
        disc(make_feats(x, x)).sum().backward()
        # (no _ladd_gside_release_disc_grads call here)
        assert not all(p.grad is None for p in disc.parameters())

    def test_discard_is_idempotent_and_latch_clears(self):
        t = self._armed_trainer(True)
        t._ladd_gside_disc_grads_dirty = True
        x = torch.randn(2, 5, 8)
        t.r3gan_disc(make_feats(x, x)).sum().backward()
        t._ladd_gside_release_disc_grads({})
        assert t._ladd_gside_disc_grads_dirty is False
        out2 = {}
        t._ladd_gside_release_disc_grads(out2)
        assert out2 == {}

    def test_discard_survives_a_backward_that_never_happened(self):
        """Latch set but the gen backward was DDP-skipped (``gen_backward_
        skipped=1``): the helper must not raise and must clear the latch."""
        t = self._armed_trainer(True)
        t._ladd_gside_disc_grads_dirty = True
        out = {}
        t._ladd_gside_release_disc_grads(out)
        assert out["train/ladd_gside_disc_grad_params"] == 0.0
        assert out["train/ladd_gside_disc_grad_norm"] == 0.0
        assert t._ladd_gside_disc_grads_dirty is False


class TestFix1Wiring:
    """The helpers above are useless unless the real call sites use them."""

    def test_both_gen_side_windows_are_flag_guarded(self):
        src = inspect.getsource(Trainer._ladd_run_pair_mode)
        # Every disc freeze on the gen side must now sit in the ``else`` of
        # a ``_ladd_gside_ckpt_recover()`` decision.
        assert src.count("self._ladd_gside_ckpt_recover()") == 2, (
            "expected the matched AND the positional gen-side window to "
            "consult the flag")
        assert src.count("disc_for_guidance.requires_grad_(False)") == 2
        assert src.count("_gsr_m") >= 3 and src.count("_gsr_p") >= 3
        for guard in ("if _gsr_m:", "if _gsr_p:",
                      "if not _gsr_m:", "if not _gsr_p:"):
            assert guard in src, f"missing guard {guard!r}"

    def test_every_generator_backward_site_discards(self):
        """One discard per ``generator_loss.backward()``. A backward site
        without one is exactly how the leak would come back."""
        import re
        src = inspect.getsource(CAFT)
        # STATEMENTS only -- ``generator_loss.backward`` also appears in a
        # dozen comments/docstrings, which are not call sites.
        n_bwd = len(re.findall(
            r"^\s+generator_loss\.backward\(", src, flags=re.M))
        n_rel = len(re.findall(
            r"^\s+self\._ladd_gside_release_disc_grads\(", src, flags=re.M))
        assert n_bwd == 3, f"unexpected backward-site count {n_bwd}"
        assert n_rel >= n_bwd, (
            f"{n_bwd} generator_loss.backward() sites but only {n_rel} "
            "discard calls")

    def test_the_disc_the_gen_side_uses_is_not_the_ddp_wrapper(self):
        """No stray allreduce: the gen-side forward must use the RAW module,
        or leaving requires_grad on would arm the DDP reducer."""
        src = inspect.getsource(Trainer._ladd_run_pair_mode)
        assert "disc_for_guidance = self.r3gan_disc\n" in src
        assert "disc_for_guidance = self.r3gan_disc_ddp" not in src

    def test_disc_update_loops_still_zero_grad_first(self):
        """The BACKSTOP the ordering proof leans on: both D-loops open with
        ``zero_grad(set_to_none=True)``, so even a missed discard cannot put
        a generator gradient into ``r3gan_optimizer.step()``."""
        src = inspect.getsource(Trainer._ladd_run_pair_mode)
        assert src.count(
            "self.r3gan_optimizer.zero_grad(set_to_none=True)") == 2


# ===========================================================================
# FIX-2 -- the action-conditioning defect
# ===========================================================================
NPB = 3
NEW_FRAMES = 9          # dmd_42f rolling: 9 new frames per roll
CF_STATE = 21


def noisy_start_for_roll(roll):
    """``current_length - new_frames - overlap`` for roll 1..R.

    Roll 1 starts at the seed boundary (0); every subsequent roll advances
    the noisy window by ``new_frames``, which is the 9*(roll-1) lag the
    diagnosis names."""
    return NEW_FRAMES * (roll - 1)


class TestFix2ActionOrigin:

    def _t(self, chunk_lo, **cfg):
        t = make_trainer(**cfg)
        t.model.streaming_state = {}
        return t, {"last_chunk_lo_in_ride_window": chunk_lo}

    @pytest.mark.parametrize("roll", [1, 2, 3, 4])
    def test_default_off_reproduces_the_legacy_lag(self, roll):
        """DEFAULT-OFF: origin is ``cf_state`` and the actions lag the
        latents by exactly ``noisy_start_sdn`` == 9*(roll-1)."""
        lag = noisy_start_for_roll(roll)
        t, s = self._t(CF_STATE + lag)
        act_lo, align = t._ladd_action_origin(CF_STATE, None, s, NPB)
        assert act_lo == CF_STATE
        assert align["lag"] == float(lag)
        assert align["lat_lo"] == float(CF_STATE + lag)
        assert align["tracking"] == 0.0

    def test_mutation_control_legacy_lag_is_not_trivially_zero(self):
        """MUTATION CONTROL: rolls 2..4 must have a NON-zero legacy lag, or
        the parametrised test above would pass even if the fix were a
        no-op."""
        lags = []
        for roll in (2, 3, 4):
            t, s = self._t(CF_STATE + noisy_start_for_roll(roll))
            _, align = t._ladd_action_origin(CF_STATE, None, s, NPB)
            lags.append(align["lag"])
        assert lags == [9.0, 18.0, 27.0]

    @pytest.mark.parametrize("roll", [1, 2, 3, 4])
    def test_flag_on_aligns_the_origin_to_the_latent_slice(self, roll):
        lat_lo = CF_STATE + noisy_start_for_roll(roll)
        t, s = self._t(lat_lo, ladd_action_origin_track_slice=True)
        act_lo, align = t._ladd_action_origin(CF_STATE, None, s, NPB)
        assert act_lo == lat_lo
        assert align["lag"] == 0.0
        assert align["tracking"] == 1.0

    def test_dmd_band_path_is_untouched_in_both_states(self):
        """``ladd_fake_sample_source='dmd'`` already passes the band's own
        absolute origin; the flag must not perturb it."""
        band_lo = 57
        for flag in (False, True):
            t, s = self._t(CF_STATE, ladd_action_origin_track_slice=flag)
            act_lo, align = t._ladd_action_origin(CF_STATE, band_lo, s, NPB)
            assert act_lo == band_lo
            assert align["lag"] == 0.0
            assert align["tracking"] == 1.0

    def test_missing_stash_falls_back_to_the_legacy_origin(self):
        """No ``last_chunk_lo_in_ride_window`` (non-streaming path): the flag
        must degrade to the legacy behaviour, never to a wrong index."""
        t = make_trainer(ladd_action_origin_track_slice=True)
        act_lo, align = t._ladd_action_origin(CF_STATE, None, {}, NPB)
        assert act_lo == CF_STATE
        assert align["lat_lo"] != align["lat_lo"]        # NaN

    @pytest.mark.parametrize("roll", [1, 2, 3, 4])
    def test_action_rows_line_up_with_the_latent_frames(self, roll):
        """END-TO-END ARITHMETIC on the real slicing rule.

        ``ride_actions`` row ``r`` is stamped with the value ``r``, so the
        action slice the disc receives for chunk ``j`` can be compared
        directly against the absolute ride-window frames that chunk covers.
        """
        lat_lo = CF_STATE + noisy_start_for_roll(roll)
        n_chunks = 6
        ride_actions = torch.arange(200, dtype=torch.float32).view(1, 200, 1)

        def slices_for(track):
            t, s = self._t(lat_lo, ladd_action_origin_track_slice=track)
            act_lo0, _ = t._ladd_action_origin(CF_STATE, None, s, NPB)
            # exactly the rule at the call site: lo_f = act_lo0 + j*npb
            return [
                ride_actions[:, act_lo0 + j * NPB: act_lo0 + (j + 1) * NPB]
                for j in range(n_chunks)
            ]

        want = [
            torch.arange(lat_lo + j * NPB, lat_lo + (j + 1) * NPB,
                         dtype=torch.float32).view(1, NPB, 1)
            for j in range(n_chunks)
        ]
        got_on = slices_for(True)
        for j in range(n_chunks):
            assert torch.equal(got_on[j], want[j]), (
                f"chunk {j}: action rows {got_on[j].flatten().tolist()} do "
                f"not cover latent frames {want[j].flatten().tolist()}")

        got_off = slices_for(False)
        if noisy_start_for_roll(roll) == 0:
            for j in range(n_chunks):
                assert torch.equal(got_off[j], want[j])
        else:
            assert not any(
                torch.equal(got_off[j], want[j]) for j in range(n_chunks)), (
                "MUTATION CONTROL: with the flag OFF the slices must be "
                "WRONG on rolls >= 2, otherwise the ON test proves nothing")


class TestFix2Telemetry:

    def _t(self, **cfg):
        t = make_trainer(**cfg)
        t._ladd_action_align = {
            "act_lo": 21.0, "lat_lo": 30.0, "lag": 9.0,
            "tracking": 0.0, "npb": 3.0,
        }
        return t

    def test_no_keys_by_default(self):
        assert self._t()._ladd_action_align_logs() == {}

    def test_log_flag_alone_measures_the_lag_without_fixing_it(self):
        logs = self._t(ladd_action_origin_log=True)._ladd_action_align_logs()
        assert logs["train/ladd_act_lo"] == 21.0
        assert logs["train/ladd_lat_lo"] == 30.0
        assert logs["train/ladd_act_lag"] == 9.0
        assert logs["train/ladd_act_lag_chunks"] == 3.0
        assert logs["train/ladd_act_origin_tracking"] == 0.0

    def test_track_flag_implies_the_telemetry(self):
        logs = self._t(
            ladd_action_origin_track_slice=True)._ladd_action_align_logs()
        assert "train/ladd_act_lag" in logs

    def test_no_stash_no_keys(self):
        t = make_trainer(ladd_action_origin_log=True)
        t._ladd_action_align = None
        assert t._ladd_action_align_logs() == {}


# ===========================================================================
# FIX-3 -- R1 telemetry + duty cycle
# ===========================================================================
HEALTHY = {
    "train/r3gan_d_real_gtxn": 0.12,
    "train/r3gan_d_fake_detached_gtxn": -0.08,
    "train/r3gan_d_loss_gtxn": 0.7,
    "train/r3gan_r1_gtxn": 0.0,               # the structurally-blind column
    "train/r3gan_disc_skipped_gtxn": 0.0,
    "train/r3gan_r1_fire_rate_gtxn": 0.5454545,
    "train/r3gan_r1_fired_total_gtxn": 30.0,
    "train/ladd_disc_d_metrics_hold_step_gtxn": 25.0,
}
DEAD = dict(HEALTHY)
DEAD["train/r3gan_r1_fire_rate_gtxn"] = 0.0
DEAD["train/r3gan_r1_fired_total_gtxn"] = 0.0


class TestFix3HealthLine:

    def test_default_line_is_byte_identical(self):
        """DEFAULT-OFF BYTE-IDENTITY: the rendered line must be exactly what
        the pre-change renderer produced, field-for-field."""
        line = CAFT.gan_health_line(HEALTHY, 31)
        assert line == (
            "[GAN-HEALTH] step=31 d_real=+0.1200 d_fake=-0.0800 "
            "d_loss=0.7000 r1=0 ratio=n/a cos=n/a disc_skipped=0 "
            "backbone_grad_norm=n/a")

    def test_mutation_control_the_default_line_would_notice_a_new_field(self):
        on = CAFT.gan_health_line(HEALTHY, 31, show_r1_fire_rate=True)
        off = CAFT.gan_health_line(HEALTHY, 31)
        assert on != off and on.startswith(off), (
            "the gated fields must be strictly APPENDED, so an existing "
            "parser reading the default line is unaffected")

    def test_r1_rate_separates_healthy_from_dead(self):
        """The whole point: ``r1=0`` is ambiguous; ``r1_rate`` is not."""
        h = CAFT.gan_health_line(HEALTHY, 31, show_r1_fire_rate=True)
        d = CAFT.gan_health_line(DEAD, 31, show_r1_fire_rate=True)
        assert " r1=0 " in h and " r1=0 " in d
        assert "r1_rate=0.545 r1_fires=30" in h
        assert "r1_rate=0.000 r1_fires=0" in d
        assert h != d

    def test_expected_live_value_for_every_n_2(self):
        """The queued arms' schedule: disc_start=20, ratio=5, N=2 =>
        55 D-updates, 30 R1 applications, rate 0.545."""
        d_updates = sum(5 for s in range(20, 71, 5))
        r1_apps = sum(5 for s in range(20, 71, 5) if s % 2 == 0)
        assert (d_updates, r1_apps) == (55, 30)
        rate = r1_apps / d_updates
        assert f"{rate:.3f}" == "0.545"
        assert "r1_rate=0.545" in CAFT.gan_health_line(
            {"train/r3gan_r1_fire_rate": rate}, 31, show_r1_fire_rate=True)

    def test_hold_step_makes_the_deferred_lag_self_describing(self):
        line = CAFT.gan_health_line(HEALTHY, 31, show_hold_step=True)
        assert "hold_step=25" in line, (
            "the label-31 line describes the step-25 D-update; without this "
            "field the 5-step lag is invisible")

    def test_absent_counters_render_n_a_not_zero(self):
        line = CAFT.gan_health_line(
            {"train/r3gan_d_loss": 1.0}, 5,
            show_r1_fire_rate=True, show_hold_step=True)
        assert "r1_rate=n/a" in line and "r1_fires=n/a" in line
        assert "hold_step=n/a" in line


class TestFix3DispatchHoldStep:

    def _t(self, **cfg):
        t = make_trainer(**cfg)
        t.step = 30
        t._ladd_d_hold = {"gt_transition": {
            "d_loss": 0.7, "d_real": 0.12, "d_fake": -0.08,
            "d_loss_stat": 0.0, "r1": 1e-9, "r1_grad_sq": 2e-9,
            "r1_fired": 1.0, "r1_block_fires": 5.0, "step": 25,
        }}
        return t

    def test_default_off_publishes_no_hold_step(self):
        out = self._t()._ladd_disc_dispatch_logs("gt_transition", True)
        assert "train/ladd_disc_d_metrics_hold_step" not in out
        assert out["train/ladd_disc_d_metrics_lagged"] == 1.0
        assert out["train/r3gan_r1"] == 1e-9      # MUTATION CONTROL: overlay
        # still works exactly as before

    def test_flag_on_publishes_the_hold_step_and_the_lag(self):
        out = self._t(
            gan_health_show_hold_step=True,
        )._ladd_disc_dispatch_logs("gt_transition", True)
        assert out["train/ladd_disc_d_metrics_hold_step"] == 25.0
        assert out["train/ladd_disc_d_metrics_hold_lag"] == 5.0

    def test_inline_dispatch_has_no_hold_and_no_keys(self):
        out = self._t(
            gan_health_show_hold_step=True,
        )._ladd_disc_dispatch_logs("gt_transition", False)
        assert out["train/ladd_disc_update_deferred"] == 0.0
        assert "train/ladd_disc_d_metrics_hold_step" not in out


class TestFix3CadenceContract:
    """FIX-3(b) is config-only; this pins the arithmetic the report cites."""

    def test_legacy_positional_modulo_fires_every_d_update_at_n_1(self):
        """``ladd_r1_unified_cadence=false`` evaluates the legacy per-iter
        expression ``_do_r1 = current_step % N == 0`` verbatim, so N=1 makes
        R1 due on EVERY D-step, i.e. on all ``gan_updates_per_step``
        D-updates of that step (the S-of-S row of the cadence contract's
        fire-rate table)."""
        d_steps = list(range(20, 71, 5))          # disc_start=20, ratio=5
        for n, expected in ((1, 11), (2, 6), (3, 3), (4, 3)):
            due = [s for s in d_steps if s % n == 0]
            assert len(due) == expected, (n, due)
        assert len([s for s in d_steps if s % 1 == 0]) == len(d_steps)
        # cost: full duty cycle turns 30 R1 applications into 55.
        assert 11 * 5 == 55 and 6 * 5 == 30


# ===========================================================================
# FIX-4 -- match pool: log the distance, raise the cap
# ===========================================================================
class TestFix4MatchDistance:

    @staticmethod
    def _stats(top_val, pool_stat, sel_pos, **kw):
        kw.setdefault("n_cand", 22)
        kw.setdefault("pool_m", len(top_val[0][0]))
        kw.setdefault("n_sel_slots", len(sel_pos[0][0]))
        kw.setdefault("cap_remaps", 0)
        kw.setdefault("n_uniq_real", 6)
        return Trainer._ladd_match_dist_stats(
            top_val, pool_stat, sel_pos, 1, 1, **kw)

    def test_perfect_match_ratio_is_far_below_one(self):
        """A working matcher: the picks are the nearest candidates, so
        chosen << pool median."""
        top_val = [[[0.1, 0.2, 0.3, 0.4]]]
        pool_stat = [[(0.1, 1.0, 1.1, 3.0)]]     # (min, median, mean, max)
        sel_pos = [[[0, 1]]]
        s = self._stats(top_val, pool_stat, sel_pos)
        assert s["ladd_match_dist_chosen"] == pytest.approx(0.15)
        assert s["ladd_match_dist_pool_median"] == 1.0
        assert s["ladd_match_dist_ratio"] == pytest.approx(0.15)
        assert s["ladd_match_dist_pool_best"] == 0.1
        assert s["ladd_match_dist_topm_worst"] == 0.4

    def test_degenerate_pool_ratio_is_one(self):
        """THE F3 FAILURE MODE: when the pool is so small that the top-M IS
        the pool, 'matching' degenerates to a random draw and the ratio
        pins at 1.0 -- which is exactly what this telemetry exists to make
        visible."""
        top_val = [[[1.0, 1.0, 1.0, 1.0]]]
        pool_stat = [[(1.0, 1.0, 1.0, 1.0)]]
        sel_pos = [[[0, 3]]]
        s = self._stats(top_val, pool_stat, sel_pos)
        assert s["ladd_match_dist_ratio"] == pytest.approx(1.0)

    def test_mutation_control_the_two_regimes_are_distinguishable(self):
        good = self._stats([[[0.1, 0.2]]], [[(0.1, 1.0, 1.0, 2.0)]], [[[0]]])
        bad = self._stats([[[1.0, 1.0]]], [[(1.0, 1.0, 1.0, 1.0)]], [[[0]]])
        assert good["ladd_match_dist_ratio"] < 0.5 < bad[
            "ladd_match_dist_ratio"]

    def test_averages_over_queries(self):
        top_val = [[[0.0, 1.0]], [[2.0, 3.0]]]
        pool_stat = [[(0.0, 1.0, 1.0, 2.0)], [(2.0, 3.0, 3.0, 4.0)]]
        sel_pos = [[[0]], [[0]]]
        s = Trainer._ladd_match_dist_stats(
            top_val, pool_stat, sel_pos, 2, 1,
            n_cand=22, pool_m=2, n_sel_slots=1, cap_remaps=0, n_uniq_real=2)
        assert s["ladd_match_dist_chosen"] == pytest.approx(1.0)
        assert s["ladd_match_dist_pool_median"] == pytest.approx(2.0)
        assert s["ladd_match_dist_n_chosen"] == 2.0

    def test_cap_remap_fraction_is_reported(self):
        s = self._stats([[[0.1, 0.2, 0.3]]], [[(0.1, 1.0, 1.0, 2.0)]],
                        [[[0, 1, 2]]], cap_remaps=2)
        assert s["ladd_match_cap_remaps"] == 2.0
        assert s["ladd_match_cap_remap_frac"] == pytest.approx(2.0 / 3.0)

    def test_logs_helper_prefixes_and_gates(self):
        t = make_trainer()
        t._ladd_match_dist = None
        assert t._ladd_match_dist_logs() == {}
        t._ladd_match_dist = {"ladd_match_dist_ratio": 0.4}
        assert t._ladd_match_dist_logs() == {
            "train/ladd_match_dist_ratio": 0.4}

    def test_matcher_only_collects_distances_when_the_flag_is_on(self):
        """DEFAULT-OFF: the collection arrays are not even allocated, so the
        matcher's hot loop is untouched."""
        src = inspect.getsource(Trainer._ladd_run_pair_mode)
        assert 'self.config, "ladd_match_distance_log"' in src
        assert "if _mdist_log else None" in src
        assert "if _mdist_log and top_val is not None:" in src


class TestFix4PoolCap:

    NPB = 3
    LEGACY_WINDOW = 69          # cf_dmdctx + actual_cap frames in the arms

    def _cand(self, frames, chunks_per_pair=2):
        return (frames // self.NPB) - chunks_per_pair + 1

    def test_the_legacy_window_really_is_the_22_candidate_clamp(self):
        """Pins finding F3's number to the code path that produces it."""
        assert self._cand(self.LEGACY_WINDOW) == 22

    def test_default_off_returns_the_legacy_window_unchanged(self):
        t = make_trainer()
        assert t._ladd_match_pool_frames(
            self.LEGACY_WINDOW, self.NPB) == self.LEGACY_WINDOW

    def test_mutation_control_default_is_not_accidentally_widened(self):
        t = make_trainer(ladd_gt_transition_match_pool_cap=0)
        for base in (30, 69, 300):
            assert t._ladd_match_pool_frames(base, self.NPB) == base

    @pytest.mark.parametrize("cap", [64, 128, 256])
    def test_raising_the_cap_widens_the_window_to_that_many_candidates(
            self, cap):
        t = make_trainer(ladd_gt_transition_match_pool_cap=cap)
        frames = t._ladd_match_pool_frames(self.LEGACY_WINDOW, self.NPB)
        assert self._cand(frames) >= cap

    def test_cap_never_shrinks_a_wider_window(self):
        t = make_trainer(ladd_gt_transition_match_pool_cap=8)
        assert t._ladd_match_pool_frames(600, self.NPB) == 600

    def test_model_attr_fallback(self):
        t = make_trainer()
        t.model.ladd_gt_transition_match_pool_cap = 64
        assert self._cand(
            t._ladd_match_pool_frames(self.LEGACY_WINDOW, self.NPB)) >= 64

    def test_the_cap_does_not_touch_the_disc_row_budget(self):
        """The reason raising it is cheap: the rows the disc forwards are
        bounded by ``ladd_gt_transition_match_max_real``, which lives in a
        different expression entirely."""
        src = inspect.getsource(Trainer._ladd_run_pair_mode)
        assert 'ladd_gt_transition_match_max_real' in src
        assert 'ladd_gt_transition_match_pool_cap' not in src


# ===========================================================================
# FIX-5 -- backbone grad-scale audit
# ===========================================================================
class TestFix5BackboneScale:

    def _t(self, *, scale, updates, trainable=True, **cfg):
        t = make_trainer(**cfg)
        t.model.ladd_fake_backbone_grad_scale = scale
        t.gan_updates_per_step = updates
        t.ladd_fake_backbone_trainable = trainable
        return t

    def test_the_arithmetic_the_claim_rests_on(self):
        """One window at the deferred site spans EVERY pending closure, i.e.
        n_modes x gan_updates_per_step disc backwards, and the backbone
        accumulates their SUM into one ``.grad``. Micro-batch groups do NOT
        multiply (each group's loss is pre-divided to sum to one D-update).
        """
        for n_modes, s in ((1, 5), (2, 5), (3, 4)):
            assert 1.0 / (n_modes * s) == pytest.approx(
                1.0 / float(n_modes * s))
        assert 1.0 / (1 * 5) == pytest.approx(0.2)
        assert 1.0 / (2 * 5) == pytest.approx(0.1)

    def test_off_by_default_says_nothing(self, caplog):
        t = self._t(scale=0.2, updates=5)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)
        assert "[LADD-BBSCALE]" not in caplog.text

    def test_one_mode_at_0_2_is_correct(self, caplog):
        t = self._t(scale=0.2, updates=5, ladd_backbone_grad_scale_check=True)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(1)
        assert "-- OK" in caplog.text
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_two_modes_at_0_2_is_flagged_as_a_double_kick(self, caplog):
        t = self._t(scale=0.2, updates=5, ladd_backbone_grad_scale_check=True)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)
        assert "MISMATCH" in caplog.text
        assert any(r.levelno >= logging.ERROR for r in caplog.records)
        assert "over-kick factor 2x" in caplog.text

    def test_gansig_of_style_arm_is_flagged(self, caplog):
        """sbatch/gansig_of.sbatch: 2 pair modes x 5 updates but scale=0.25
        => 2.5x the intended adversarial pull on the critic backbone."""
        t = self._t(scale=0.25, updates=5, ladd_backbone_grad_scale_check=True)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)
        assert "MISMATCH" in caplog.text
        assert "2.5x" in caplog.text

    def test_gansig_real_style_arm_is_correct(self, caplog):
        t = self._t(scale=0.1, updates=5, ladd_backbone_grad_scale_check=True)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)
        assert "-- OK" in caplog.text

    def test_never_auto_corrects(self, caplog):
        t = self._t(scale=0.2, updates=5, ladd_backbone_grad_scale_check=True)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)
        assert t.model.ladd_fake_backbone_grad_scale == 0.2, (
            "the check must be REPORT-ONLY -- silently fixing it would hide "
            "the mis-specification from the researcher")

    def test_inert_when_the_backbone_is_frozen(self, caplog):
        """The scale hook is not installed at all on the frozen path, so a
        'wrong' value there is meaningless and must not raise an alarm."""
        t = self._t(scale=0.2, updates=5, trainable=False,
                    ladd_backbone_grad_scale_check=True)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)
        assert "[LADD-BBSCALE]" not in caplog.text

    def test_runs_only_once(self, caplog):
        t = self._t(scale=0.2, updates=5, ladd_backbone_grad_scale_check=True)
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)
            t._ladd_check_backbone_grad_scale(2)
        assert caplog.text.count("[LADD-BBSCALE]") == 1

    def test_never_takes_training_down(self, caplog):
        t = self._t(scale=0.2, updates=5, ladd_backbone_grad_scale_check=True)
        t.gan_updates_per_step = "not-an-int"
        with caplog.at_level(logging.INFO):
            t._ladd_check_backbone_grad_scale(2)      # must not raise

    def test_the_probe_really_multiplies_the_incoming_gradient(self):
        """Ground the claim in the mechanism: ``_install_grad_probes``
        installs a ``register_hook`` that REPLACES the gradient with
        ``g * scale``."""
        from model import dmd_action_forcing as DAF
        src = inspect.getsource(DAF._LaddFakeFeatureBackbone)
        assert "def _probe(g):" in src
        assert "g = g * _s" in src
        assert "return g if do_scale else None" in src
