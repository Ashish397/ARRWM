"""A22 held-out discriminator generalisation probe — unit tests.

Covers the pure, GPU-free core:
  * the interpretation table (``classify`` / ``probe_statistics``),
  * the holdout leak guards (static set check + runtime supply tripwire),
  * the coarse vertical band plan (A24),
  * the default-off / byte-identical contract of the entry point.

Run:
    PYTHONPATH=.:action-forcing pytest -q testing/test_disc_holdout_probe.py
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.disc_holdout_probe import (  # noqa: E402
    DEFAULTS,
    HoldoutLeakError,
    PREFIX,
    VERDICT_BOTH_CHANCE,
    VERDICT_D_UNDERPOWERED,
    VERDICT_GENERALISING,
    VERDICT_INVERTED,
    VERDICT_MEMORISING,
    VERDICT_STUDENT_HARD,
    auc_null_se,
    band_plan,
    classify,
    holdout_leak_check,
    maybe_run_holdout_probe,
    note_real_supply_ride,
    probe_statistics,
    register_fake_source,
    register_score_fn,
    ride_key,
    roc_auc,
    run_leak_check,
    take_crops,
)


def _fake_trainer(**cfg_kwargs):
    """Minimal trainer stand-in: only the attributes the probe reads."""
    cfg = types.SimpleNamespace(**cfg_kwargs)
    tr = types.SimpleNamespace()
    tr.config = cfg
    tr.step = 0
    tr.is_main_process = True
    # Non-empty by default: an empty training ride list is now a FAIL-CLOSED
    # condition (D3), not a benign stub state. Tests that want the failure ask
    # for it explicitly.
    tr.dataset = types.SimpleNamespace(
        _rides=[f"/enc/train{i}.zarr" for i in range(8)])
    tr._dhp_holdout_paths = [f"/held/h{i}.zarr" for i in range(8)]
    return tr


class TestRocAuc(unittest.TestCase):
    def test_perfect_and_reversed_separation(self):
        pos = torch.tensor([3.0, 4.0, 5.0])
        neg = torch.tensor([0.0, 1.0, 2.0])
        self.assertAlmostEqual(roc_auc(pos, neg), 1.0, places=6)
        self.assertAlmostEqual(roc_auc(neg, pos), 0.0, places=6)

    def test_ties_score_half(self):
        a = torch.tensor([1.0, 1.0])
        self.assertAlmostEqual(roc_auc(a, a), 0.5, places=6)

    def test_null_se_shrinks_with_n(self):
        self.assertGreater(auc_null_se(8, 8), auc_null_se(256, 256))


class TestInterpretationTable(unittest.TestCase):
    """Every row of the module docstring's table must be reachable."""

    def test_both_up_is_generalising(self):
        self.assertEqual(
            classify(0.95, 0.90, se=0.01, chance_band=0.05),
            VERDICT_GENERALISING,
        )

    def test_train_up_heldout_chance_is_memorisation(self):
        self.assertEqual(
            classify(0.95, 0.50, se=0.01, chance_band=0.05),
            VERDICT_MEMORISING,
        )

    def test_both_chance_without_control_is_undecidable(self):
        self.assertEqual(
            classify(0.51, 0.49, se=0.01, chance_band=0.05),
            VERDICT_BOTH_CHANCE,
        )

    def test_both_chance_splits_once_the_a14_control_is_supplied(self):
        self.assertEqual(
            classify(0.51, 0.49, se=0.01, chance_band=0.05, control_auc=0.99),
            VERDICT_STUDENT_HARD,
        )
        self.assertEqual(
            classify(0.51, 0.49, se=0.01, chance_band=0.05, control_auc=0.52),
            VERDICT_D_UNDERPOWERED,
        )

    def test_heldout_up_train_chance_is_flagged_as_a_build_bug(self):
        self.assertEqual(
            classify(0.50, 0.95, se=0.01, chance_band=0.05),
            VERDICT_INVERTED,
        )

    def test_small_sample_noise_cannot_manufacture_a_verdict(self):
        # 0.62 with a huge null SE must NOT read as separation.
        self.assertEqual(
            classify(0.62, 0.50, se=0.10, chance_band=0.02),
            VERDICT_BOTH_CHANCE,
        )


class TestProbeStatistics(unittest.TestCase):
    def test_memorisation_signature(self):
        """D scores its own training reals high, unseen reals like fakes."""
        real_train = torch.linspace(4.0, 6.0, 24)
        real_heldout = torch.linspace(-1.0, 1.0, 24)
        fake = torch.linspace(-1.0, 1.0, 24)
        st = probe_statistics(real_train, real_heldout, fake)
        self.assertAlmostEqual(st["train_acc"], 1.0, places=6)
        self.assertAlmostEqual(st["heldout_acc"], 0.5, places=2)
        self.assertGreater(st["gen_gap"], 3.0)
        self.assertGreater(st["real_gap"], 3.0)
        self.assertAlmostEqual(st["real_auc"], 1.0, places=6)
        self.assertEqual(st["verdict"], VERDICT_MEMORISING)

    def test_generalising_signature(self):
        real_train = torch.linspace(4.0, 6.0, 24)
        real_heldout = torch.linspace(3.8, 5.8, 24)
        fake = torch.linspace(-1.0, 1.0, 24)
        st = probe_statistics(real_train, real_heldout, fake)
        self.assertEqual(st["verdict"], VERDICT_GENERALISING)
        self.assertLess(abs(st["gen_gap"]), 0.5)
        self.assertLess(abs(st["real_auc"] - 0.5), 0.2)

    def test_fake_side_is_shared_by_both_comparisons(self):
        """train_margin - heldout_margin must equal the pure real-side gap:
        that identity is what makes the two margins comparable."""
        rt = torch.randn(16) + 2.0
        rh = torch.randn(16) + 1.0
        fk = torch.randn(16)
        st = probe_statistics(rt, rh, fk)
        self.assertAlmostEqual(st["gen_gap"], st["real_gap"], places=6)

    def test_counts_are_reported(self):
        st = probe_statistics(torch.randn(5), torch.randn(7), torch.randn(9))
        self.assertEqual(st["n_real_train"], 5.0)
        self.assertEqual(st["n_real_heldout"], 7.0)
        self.assertEqual(st["n_fake"], 9.0)


class TestHoldoutIntegrity(unittest.TestCase):
    def test_ride_identity_is_the_basename(self):
        self.assertEqual(ride_key("/a/b/20240306095439.zarr"), "20240306095439.zarr")

    def test_superset_root_leak_is_detected_across_different_paths(self):
        """The recorded historical failure: encoded_root switched from weu to
        its superset weunz with holdout_zarr_list=null. Path identity would
        miss it; basename identity must catch it."""
        train = [
            "/projects/fbots/frodobots_encoded_weunz/20240306095439.zarr",
            "/projects/fbots/frodobots_encoded_weunz/20240101000000.zarr",
        ]
        held = ["/projects/fbots/frodobots_encoded_weu_holdout/20240306095439.zarr"]
        rep = holdout_leak_check(train, held)
        self.assertEqual(rep["n_leak"], 1)
        self.assertEqual(rep["leaked"], ["20240306095439.zarr"])

    def test_clean_split_reports_zero_leak(self):
        train = ["/x/a.zarr", "/x/b.zarr"]
        held = ["/y/c.zarr"]
        self.assertEqual(holdout_leak_check(train, held)["n_leak"], 0)

    def test_strict_mode_raises_on_a_leak(self):
        tr = _fake_trainer(disc_holdout_probe_strict=True)
        tr.dataset._rides = [(Path("/train/a.zarr"), None, {}, 100)]
        tr._dhp_holdout_paths = [Path("/held/a.zarr")]
        with self.assertRaises(HoldoutLeakError):
            run_leak_check(tr)

    def test_non_strict_mode_reports_instead_of_raising(self):
        tr = _fake_trainer(disc_holdout_probe_strict=False)
        tr.dataset._rides = [(Path("/train/a.zarr"), None, {}, 100)]
        tr._dhp_holdout_paths = [Path("/held/a.zarr")]
        self.assertEqual(run_leak_check(tr)["n_leak"], 1)

    def test_supply_tripwire_counts_only_held_out_rides(self):
        tr = _fake_trainer()
        tr._dhp_holdout_names = {"held.zarr"}
        note_real_supply_ride(tr, "/enc/train1.zarr")
        note_real_supply_ride(tr, "/enc/train2.zarr")
        self.assertEqual(int(getattr(tr, "_dhp_leak_seen", 0)), 0)
        note_real_supply_ride(tr, "/enc/held.zarr")
        self.assertEqual(int(tr._dhp_leak_seen), 1)
        # And it records the observed training rides for the train reference.
        self.assertEqual(
            tr._dhp_observed_rides,
            ["train1.zarr", "train2.zarr", "held.zarr"],
        )

    def test_observed_ride_list_is_capped_and_deduplicated(self):
        tr = _fake_trainer(disc_holdout_probe_observed_cap=3)
        tr._dhp_holdout_names = set()
        for i in range(6):
            note_real_supply_ride(tr, f"/enc/r{i}.zarr")
        note_real_supply_ride(tr, "/enc/r5.zarr")  # duplicate
        self.assertEqual(tr._dhp_observed_rides, ["r3.zarr", "r4.zarr", "r5.zarr"])

    def test_tripwire_never_raises(self):
        tr = _fake_trainer()
        note_real_supply_ride(tr, None)  # must not blow up
        note_real_supply_ride(tr, "")


class TestBandPlan(unittest.TestCase):
    def test_offsets_stay_inside_their_band_and_in_range(self):
        g = torch.Generator(device="cpu").manual_seed(0)
        bands, offs = band_plan(200, n_rows=60, crop_rows=24, n_bands=3, gen=g)
        max_off = 60 - 24  # 36 -> bands 0-12 / 12-24 / 24-36
        for b, y in zip(bands, offs):
            self.assertGreaterEqual(y, b * max_off // 3)
            self.assertLessEqual(y, (b + 1) * max_off // 3)
            self.assertLessEqual(y + 24, 60)
        self.assertEqual(set(bands), {0, 1, 2})

    def test_bands_are_coarse_not_exact_y(self):
        """A24: band matching constrains the crop CENTROID to one of a handful
        of bins; it must not collapse to a single y per band."""
        g = torch.Generator(device="cpu").manual_seed(1)
        _, offs = band_plan(200, n_rows=60, crop_rows=24, n_bands=3, gen=g)
        self.assertGreater(len(set(offs)), 10)

    def test_degenerate_geometry_is_safe(self):
        g = torch.Generator(device="cpu").manual_seed(2)
        bands, offs = band_plan(4, n_rows=24, crop_rows=24, n_bands=3, gen=g)
        self.assertEqual(offs, [0, 0, 0, 0])
        self.assertEqual(len(bands), 4)

    def test_take_crops_shape_and_bounds(self):
        g = torch.Generator(device="cpu").manual_seed(3)
        src = torch.randn(5, 3, 16, 60, 104)
        _, offs = band_plan(8, 60, 24, 3, g)
        out = take_crops(src, offs, 24, 32, g)
        self.assertEqual(tuple(out.shape), (8, 3, 16, 24, 32))

    def test_one_band_plan_is_shared_by_all_three_sides(self):
        """The same offsets applied to fake / train-real / held-out-real is
        what makes vertical content unable to explain a train-vs-held-out gap.
        """
        g = torch.Generator(device="cpu").manual_seed(4)
        _, offs = band_plan(6, 60, 24, 3, g)
        a = take_crops(torch.zeros(2, 3, 16, 60, 104), offs, 24, 32, g)
        b = take_crops(torch.zeros(2, 3, 16, 60, 104), offs, 24, 32, g)
        self.assertEqual(a.shape, b.shape)


class TestDefaultOffContract(unittest.TestCase):
    def test_master_gate_defaults_to_off(self):
        self.assertEqual(DEFAULTS["disc_holdout_probe_every"], 0)

    def test_disabled_probe_writes_nothing(self):
        tr = _fake_trainer()
        logs = {}
        maybe_run_holdout_probe(tr, torch.zeros(1, 3, 16, 60, 104), logs)
        self.assertEqual(logs, {})

    def test_none_latents_writes_nothing(self):
        tr = _fake_trainer(disc_holdout_probe_every=1)
        logs = {}
        maybe_run_holdout_probe(tr, None, logs)
        self.assertEqual(logs, {})

    def test_enabled_without_a_critic_still_reports_holdout_integrity(self):
        """Pre-B2 state: no pixel critic exists, but the leak guards must
        already be live and readable."""
        tr = _fake_trainer(
            disc_holdout_probe_every=1, disc_holdout_probe_strict=True,
        )
        tr.dataset._rides = [(Path("/train/a.zarr"), None, {}, 100)]
        tr._dhp_holdout_paths = [Path("/held/h1.zarr"), Path("/held/h2.zarr")]
        logs = {}
        maybe_run_holdout_probe(tr, torch.zeros(1, 3, 16, 60, 104), logs)
        self.assertEqual(logs[PREFIX + "leak_rides"], 0.0)
        self.assertEqual(logs[PREFIX + "holdout_rides"], 2.0)
        self.assertEqual(logs[PREFIX + "no_critic"], 1.0)
        self.assertEqual(logs[PREFIX + "ring_delta"], 0.0)
        self.assertNotIn(PREFIX + "fired", logs)

    def test_once_per_step_latch(self):
        # Latch only: the probe must not fire twice in one step. Config is
        # VALID here (non-empty holdout + training rides) so the fail-closed
        # guard is not what is under test.
        tr = _fake_trainer(disc_holdout_probe_every=1)
        logs1, logs2 = {}, {}
        maybe_run_holdout_probe(tr, torch.zeros(1, 3, 16, 60, 104), logs1)
        maybe_run_holdout_probe(tr, torch.zeros(1, 3, 16, 60, 104), logs2)
        self.assertEqual(logs2, {}, "probe fired twice in one step")

    def test_empty_holdout_fails_closed_not_open(self):
        """D2: an unresolved reserved-ride set must NEVER read as clean.

        `holdout_eval_root` appears in no yaml -- it exists only on --override
        lines -- so unset/misspelled is the LIKELY state, and the old code
        emitted leak_rides=leak_seen=ring_delta=0 (the green signature) while
        the runtime tripwire was a permanent no-op.
        """
        import model.disc_holdout_probe as _d
        tr = _fake_trainer(disc_holdout_probe_every=1,
                           disc_holdout_probe_strict=True)
        rep = {"n_train": 8, "n_holdout": 0, "n_leak": 0, "leaked": []}
        with self.assertRaises(_d.HoldoutUnverifiableError):
            _d._fail_closed(tr, rep)
        self.assertEqual(rep["unverifiable"], 1.0)

    def test_empty_training_rides_fails_closed(self):
        """D3: an unreachable dataset._rides compared against nothing."""
        import model.disc_holdout_probe as _d
        tr = _fake_trainer(disc_holdout_probe_strict=True)
        rep = {"n_train": 0, "n_holdout": 20, "n_leak": 0, "leaked": []}
        with self.assertRaises(_d.HoldoutUnverifiableError):
            _d._fail_closed(tr, rep)

    def test_healthy_config_passes_the_guard(self):
        import model.disc_holdout_probe as _d
        tr = _fake_trainer(disc_holdout_probe_strict=True)
        rep = {"n_train": 2529, "n_holdout": 20, "n_leak": 0, "leaked": []}
        _d._fail_closed(tr, rep)                     # must not raise
        self.assertEqual(rep["unverifiable"], 0.0)

    def test_strict_leak_reraises_from_cache(self):
        """D6: strict mode was one-shot -- the report was cached BEFORE the
        raise, so any enclosing retry turned a hard fail into a silent
        leak_rides=1 on a curve nobody watches."""
        import model.disc_holdout_probe as _d
        tr = _fake_trainer(disc_holdout_probe_strict=True)
        tr._dhp_leak_report = {"n_train": 2529, "n_holdout": 20, "n_leak": 1,
                               "leaked": ["x"], "_msg": "leak"}
        for _ in range(3):
            with self.assertRaises(_d.HoldoutLeakError):
                _d.run_leak_check(tr)

    def test_registered_score_fn_takes_precedence(self):
        tr = _fake_trainer()
        tr.pixel_texture_disc = lambda px: torch.zeros(px.shape[0])
        register_score_fn(tr, lambda px: torch.ones(px.shape[0]))
        from model.disc_holdout_probe import _resolve_score_fn
        fn = _resolve_score_fn(tr)
        self.assertTrue(bool(fn(torch.zeros(2, 3, 8, 8)).eq(1.0).all()))

    def test_b2_attribute_is_discovered_with_no_registration(self):
        tr = _fake_trainer()
        tr.pixel_texture_disc = lambda px: torch.full((px.shape[0], 1, 4, 4), 2.0)
        from model.disc_holdout_probe import _resolve_score_fn, _reduce_scores
        fn = _resolve_score_fn(tr)
        self.assertIsNotNone(fn)
        self.assertTrue(
            bool(_reduce_scores(fn(torch.zeros(3, 3, 8, 8))).eq(2.0).all())
        )

    def test_probe_end_to_end_with_a_stub_critic(self):
        """Full path with the zarr pools stubbed: exercises band plan ->
        crop -> decode -> score -> statistics -> telemetry keys."""
        tr = _fake_trainer(
            disc_holdout_probe_every=1,
            disc_holdout_probe_crops=6,
            disc_holdout_probe_frames=2,
            disc_holdout_probe_crop_lat=[8, 8],
            disc_holdout_probe_border_px=0,
        )
        tr._dhp_holdout_paths = [Path("/held/h1.zarr")]
        tr._dhp_heldout_pool = torch.randn(4, 3, 16, 60, 104)
        tr._dhp_train_pool = torch.randn(4, 3, 16, 60, 104) + 5.0
        tr._ladd_real_ring = [1, 2, 3]

        # Stand-in VAE decode: latent [n, F, C, h, w] -> pixels [n, 4F, 3, ...]
        def _decode(lat):
            n, f, _c, h, w = lat.shape
            base = lat[:, :, :3].mean(dim=2, keepdim=True).expand(-1, -1, 3, -1, -1)
            return base.repeat_interleave(4, dim=1).reshape(n, 4 * f, 3, h, w)

        tr._vae_decode_nograd = _decode
        # Critic that simply reports mean brightness -> train reals score high.
        register_score_fn(tr, lambda px: px.reshape(px.shape[0], -1).mean(dim=1))

        logs = {}
        maybe_run_holdout_probe(tr, torch.randn(1, 3, 16, 60, 104), logs)
        self.assertEqual(logs.get(PREFIX + "err", 0.0), 0.0, msg=str(logs))
        self.assertEqual(logs[PREFIX + "fired"], 1.0)
        self.assertEqual(logs[PREFIX + "ring_delta"], 0.0)
        self.assertEqual(logs[PREFIX + "band_mismatch"], 0.0)
        self.assertEqual(logs[PREFIX + "n_real_train"], 12.0)
        self.assertEqual(logs[PREFIX + "n_real_heldout"], 12.0)
        self.assertEqual(logs[PREFIX + "n_fake"], 12.0)
        # Train reals were offset by +5 -> the memorisation signature.
        self.assertGreater(logs[PREFIX + "real_gap"], 1.0)
        self.assertEqual(logs[PREFIX + "verdict"], VERDICT_MEMORISING)
        self.assertEqual(len(tr._ladd_real_ring), 3)

    def test_registered_fake_source_overrides_the_default_tensor(self):
        """A2/A23: the probe must score the tensor B2's critic is trained on,
        and must SAY SO when it falls back to the trainer's default."""
        tr = _fake_trainer(
            disc_holdout_probe_every=1,
            disc_holdout_probe_crops=4,
            disc_holdout_probe_frames=1,
            disc_holdout_probe_crop_lat=[8, 8],
            disc_holdout_probe_border_px=0,
        )
        tr._dhp_holdout_paths = [Path("/held/h1.zarr")]
        tr._dhp_heldout_pool = torch.zeros(2, 3, 16, 60, 104)
        tr._dhp_train_pool = torch.zeros(2, 3, 16, 60, 104)

        def _decode(lat):
            n, f, _c, h, w = lat.shape
            base = lat[:, :, :3].mean(dim=2, keepdim=True).expand(-1, -1, 3, -1, -1)
            return base.repeat_interleave(4, dim=1).reshape(n, 4 * f, 3, h, w)

        tr._vae_decode_nograd = _decode
        register_score_fn(tr, lambda px: px.reshape(px.shape[0], -1).mean(dim=1))

        register_fake_source(tr, torch.full((1, 3, 16, 60, 104), -9.0))
        logs = {}
        maybe_run_holdout_probe(tr, torch.zeros(1, 3, 16, 60, 104), logs)
        self.assertEqual(logs[PREFIX + "fake_src_default"], 0.0)
        self.assertLess(logs[PREFIX + "fake_mean"], -1.0)   # the staged tensor
        self.assertIsNone(tr._dhp_fake_latents)             # consumed

        tr._dhp_last_step = None                            # next fire
        logs2 = {}
        maybe_run_holdout_probe(tr, torch.zeros(1, 3, 16, 60, 104), logs2)
        self.assertEqual(logs2[PREFIX + "fake_src_default"], 1.0)
        self.assertAlmostEqual(logs2[PREFIX + "fake_mean"], 0.0, places=5)

    def test_probe_failure_is_swallowed_not_fatal(self):
        tr = _fake_trainer(disc_holdout_probe_every=1)
        tr._dhp_holdout_paths = [Path("/held/h1.zarr")]
        tr._dhp_heldout_pool = torch.randn(2, 3, 16, 60, 104)
        tr._dhp_train_pool = torch.randn(2, 3, 16, 60, 104)

        def _boom(_lat):
            raise RuntimeError("simulated decoder OOM")

        tr._vae_decode_nograd = _boom
        register_score_fn(tr, lambda px: px.mean(dim=(1, 2, 3)))
        logs = {}
        maybe_run_holdout_probe(tr, torch.randn(1, 3, 16, 60, 104), logs)
        self.assertEqual(logs[PREFIX + "err"], 1.0)
        self.assertNotIn(PREFIX + "fired", logs)


if __name__ == "__main__":
    unittest.main()
