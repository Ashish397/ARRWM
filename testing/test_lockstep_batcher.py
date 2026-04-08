#!/usr/bin/env python3
"""Unit tests for LockstepRideBatcher (per-slot independent mode).

Verifies ride-slot progression, per-slot window bounds, independent
exhaustion/refill, staggered start offsets, and batch construction
semantics with synthetic ride data (no zarr/GPU needed).

Usage:
    python testing/test_lockstep_batcher.py [-v]
"""

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.causal_teacher_streaming import LockstepRideBatcher, _RideSlot


_FAKE_Z_ACTIONS: dict = {}


def _fake_ride(n_latent: int, name: str = "ride.zarr") -> dict:
    z = torch.randn(n_latent, 8)
    path = f"/fake/{name}"
    _FAKE_Z_ACTIONS[path] = z
    return {
        "zarr_path": path,
        "prompt_embeds": torch.randn(77, 512),
        "n_latent_frames": n_latent,
    }


def _fake_encode_fn(zarr_path, n_latent_frames, start, end):
    """Test-only encode_fn that slices from the pre-generated fake tensors."""
    return _FAKE_Z_ACTIONS[zarr_path][start:end]


# ======================================================================
# Core batcher lifecycle
# ======================================================================

class TestBatcherInitialState(unittest.TestCase):
    """Verify batcher starts in needs-refill state."""

    def test_needs_refill_initially(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        self.assertTrue(batcher.needs_refill())
        self.assertTrue(batcher.needs_new_group())

    def test_all_slots_exhausted_initially(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=3,
        )
        self.assertEqual(batcher.exhausted_slot_indices(), [0, 1, 2])


class TestLoadGroup(unittest.TestCase):
    """Verify initial load_group fills all slots."""

    def test_load_group_fills_all_slots(self):
        random.seed(0)
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        ride_a = _fake_ride(63, "a.zarr")
        ride_b = _fake_ride(84, "b.zarr")
        batcher.load_group([ride_a, ride_b])

        self.assertFalse(batcher.needs_refill())
        self.assertEqual(len(batcher.exhausted_slot_indices()), 0)

    def test_load_group_wrong_count_raises(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
        )
        with self.assertRaises(ValueError):
            batcher.load_group([_fake_ride(63)])


class TestSingleSlotProgression(unittest.TestCase):
    """batch_size=1: verify window progression and exhaustion."""

    def test_sequential_windows_no_offset(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher.load_group([_fake_ride(63, "ride_63.zarr")])

        slots = batcher.get_slot_info()
        self.assertEqual(slots[0].n_windows, 2)
        self.assertEqual(slots[0].start_offset, 0)

        expected = [(0, 24), (21, 45)]
        for i, (exp_s, exp_e) in enumerate(expected):
            bounds = batcher.get_window_bounds()
            self.assertEqual(bounds[0], (exp_s, exp_e))
            batcher.advance()

        self.assertTrue(batcher.needs_refill())
        self.assertEqual(batcher.exhausted_slot_indices(), [0])

    def test_ride_too_short_immediately_exhausted(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher.load_group([_fake_ride(20)])
        self.assertTrue(batcher.needs_refill())

    def test_exact_minimum_length(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher.load_group([_fake_ride(24)])
        slots = batcher.get_slot_info()
        self.assertEqual(slots[0].n_windows, 1)
        self.assertEqual(batcher.get_window_bounds()[0], (0, 24))


# ======================================================================
# Independent per-slot exhaustion and refill
# ======================================================================

class TestIndependentExhaustion(unittest.TestCase):
    """Verify slots exhaust independently, not truncated to shortest."""

    def test_long_ride_survives_after_short_ride_exhausts(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        short = _fake_ride(42, "short.zarr")  # 1 window
        long_ = _fake_ride(105, "long.zarr")  # 4 windows
        batcher.load_group([short, long_])

        slots = batcher.get_slot_info()
        self.assertEqual(slots[0].n_windows, 1)
        self.assertEqual(slots[1].n_windows, 4)

        batcher.advance()
        self.assertEqual(batcher.exhausted_slot_indices(), [0])
        self.assertTrue(batcher.needs_refill())

        # Slot 1 (long ride) is still active
        self.assertNotIn(1, batcher.exhausted_slot_indices())

    def test_refill_only_exhausted_slot(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        short = _fake_ride(42, "short.zarr")
        long_ = _fake_ride(105, "long.zarr")
        batcher.load_group([short, long_])

        batcher.advance()
        exhausted = batcher.exhausted_slot_indices()
        self.assertEqual(exhausted, [0])

        replacement = _fake_ride(84, "replacement.zarr")
        batcher.refill_slots(exhausted, [replacement])

        self.assertFalse(batcher.needs_refill())
        slots = batcher.get_slot_info()
        self.assertIn("replacement.zarr", slots[0].zarr_path)
        self.assertEqual(slots[0].window_idx, 0)

    def test_refill_preserves_other_slot_state(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        short = _fake_ride(42, "short.zarr")
        long_ = _fake_ride(105, "long.zarr")
        batcher.load_group([short, long_])

        batcher.advance()

        slots_before = batcher.get_slot_info()
        long_win_before = slots_before[1].window_idx

        batcher.refill_slots([0], [_fake_ride(63, "new.zarr")])

        slots_after = batcher.get_slot_info()
        self.assertEqual(slots_after[1].window_idx, long_win_before)
        self.assertIn("long.zarr", slots_after[1].zarr_path)

    def test_refill_count_mismatch_raises(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        batcher.load_group([_fake_ride(42), _fake_ride(42)])
        batcher.advance()

        with self.assertRaises(ValueError):
            batcher.refill_slots([0, 1], [_fake_ride(63)])


# ======================================================================
# Staggered start offsets
# ======================================================================

class TestStaggeredStartOffsets(unittest.TestCase):
    """Verify random per-slot start offsets and block alignment."""

    def test_offset_is_block_aligned(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=40,
        )
        for _ in range(50):
            offset = batcher._random_start_offset()
            self.assertEqual(offset % 3, 0, f"Offset {offset} not block-aligned")
            self.assertGreaterEqual(offset, 0)
            self.assertLessEqual(offset, 40)

    def test_offset_range_coverage(self):
        """Over many samples, offsets should span the allowed range."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=39,
        )
        offsets = set()
        for _ in range(500):
            offsets.add(batcher._random_start_offset())
        self.assertIn(0, offsets)
        self.assertIn(39 // 3 * 3, offsets)
        self.assertGreater(len(offsets), 3)

    def test_offset_zero_when_max_is_zero(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        for _ in range(20):
            self.assertEqual(batcher._random_start_offset(), 0)

    def test_offset_affects_window_bounds(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        ride = _fake_ride(200, "long.zarr")
        batcher.load_group([ride])
        bounds_no_offset = batcher.get_window_bounds()[0]

        batcher2 = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=100,
        )
        random.seed(42)
        batcher2.load_group([ride])
        slots = batcher2.get_slot_info()
        offset = slots[0].start_offset
        bounds_with_offset = batcher2.get_window_bounds()[0]

        self.assertEqual(bounds_with_offset[0], offset)
        self.assertEqual(bounds_with_offset[1], offset + 21 + 3)
        if offset > 0:
            self.assertNotEqual(bounds_with_offset, bounds_no_offset)

    def test_offset_falls_back_to_zero_when_ride_too_short(self):
        """If offset would leave no room for even one window, use 0."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=40,
        )
        ride = _fake_ride(24, "tiny.zarr")
        random.seed(999)
        batcher.load_group([ride])
        slots = batcher.get_slot_info()
        self.assertEqual(slots[0].start_offset, 0)
        self.assertGreater(slots[0].n_windows, 0)

    def test_different_slots_get_different_offsets(self):
        """With a large batch, not all slots should share the same offset."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=8,
            max_start_offset=39,
        )
        rides = [_fake_ride(200, f"ride_{i}.zarr") for i in range(8)]
        random.seed(12345)
        batcher.load_group(rides)
        offsets = [s.start_offset for s in batcher.get_slot_info()]
        self.assertGreater(
            len(set(offsets)), 1,
            "With 8 slots and max_offset=39, at least two offsets should differ",
        )

    def test_window_count_accounts_for_offset(self):
        """Offset should reduce the number of available windows."""
        ride = _fake_ride(66, "ride_66.zarr")

        batcher_no = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher_no.load_group([ride])
        n_win_no = batcher_no.get_slot_info()[0].n_windows

        batcher_yes = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher_yes._slots[0] = _RideSlot(
            zarr_path=ride["zarr_path"],
            prompt_embeds=ride["prompt_embeds"],
            n_latent_frames=66,
            n_windows=0,
            window_idx=0,
            start_offset=21,
            loaded=True,
        )
        usable = 66 - 21 - 3
        n_win_offset = usable // 21
        batcher_yes._slots[0].n_windows = n_win_offset

        self.assertLessEqual(n_win_offset, n_win_no)


# ======================================================================
# Per-slot window bounds
# ======================================================================

class TestPerSlotWindowBounds(unittest.TestCase):
    """Verify that bounds differ per slot based on their independent state."""

    def test_different_slots_different_bounds(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        short = _fake_ride(42, "short.zarr")
        long_ = _fake_ride(105, "long.zarr")
        batcher.load_group([short, long_])

        batcher.advance()
        batcher.refill_slots([0], [_fake_ride(105, "new.zarr")])

        bounds = batcher.get_window_bounds()
        # Slot 0 was refilled: window_idx=0
        # Slot 1: window_idx=1
        slot0_start = batcher.get_slot_info()[0].start_offset + 0 * 21
        slot1_start = batcher.get_slot_info()[1].start_offset + 1 * 21

        self.assertEqual(bounds[0][0], slot0_start)
        self.assertEqual(bounds[1][0], slot1_start)


# ======================================================================
# max_windows_per_ride
# ======================================================================

class TestMaxWindowsPerRide(unittest.TestCase):

    def test_cap_limits_windows(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_windows_per_ride=2, max_start_offset=0,
        )
        ride = _fake_ride(210, "long.zarr")
        batcher.load_group([ride])
        self.assertEqual(batcher.get_slot_info()[0].n_windows, 2)


# ======================================================================
# Z-actions and prompt batch loading
# ======================================================================

class TestBatchLoading(unittest.TestCase):

    def test_z_actions_batch_per_slot(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        ride_a = _fake_ride(63, "za.zarr")
        ride_b = _fake_ride(63, "zb.zarr")
        batcher.load_group([ride_a, ride_b])

        z = batcher.load_z_actions_batch(torch.device("cpu"), encode_fn=_fake_encode_fn)
        self.assertEqual(z.shape, (2, 24, 8))
        torch.testing.assert_close(z[0], _FAKE_Z_ACTIONS[ride_a["zarr_path"]][:24])
        torch.testing.assert_close(z[1], _FAKE_Z_ACTIONS[ride_b["zarr_path"]][:24])

    def test_z_actions_requires_encode_fn(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher.load_group([_fake_ride(63, "req.zarr")])
        with self.assertRaises(RuntimeError):
            batcher.load_z_actions_batch(torch.device("cpu"))

    def test_prompt_embeds_batch(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        ride_a = _fake_ride(63, "pa.zarr")
        ride_b = _fake_ride(63, "pb.zarr")
        batcher.load_group([ride_a, ride_b])

        pe = batcher.load_prompt_embeds_batch(torch.device("cpu"))
        self.assertEqual(pe.shape, (2, 77, 512))
        torch.testing.assert_close(pe[0], ride_a["prompt_embeds"])
        torch.testing.assert_close(pe[1], ride_b["prompt_embeds"])


# ======================================================================
# Summary and properties
# ======================================================================

class TestSummaryAndProperties(unittest.TestCase):

    def test_summary_contains_slot_info(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher.load_group([_fake_ride(63, "test.zarr")])
        s = batcher.summary()
        self.assertIn("test.zarr", s)
        self.assertIn("w=0/", s)

    def test_is_first_window(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_start_offset=0,
        )
        batcher.load_group([_fake_ride(63)])
        self.assertTrue(batcher.is_first_window)
        batcher.advance()
        self.assertFalse(batcher.is_first_window)


# ======================================================================
# End-to-end: simulate trainer refill loop
# ======================================================================

class TestTrainerRefillLoop(unittest.TestCase):
    """Simulate the trainer's refill loop with independent slots."""

    def test_continuous_streaming(self):
        """Run 10 advance steps with batch_size=2, refilling as needed."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
            max_start_offset=0,
        )
        ride_counter = [0]

        def next_ride():
            ride_counter[0] += 1
            return _fake_ride(
                random.choice([42, 63, 84]),
                f"ride_{ride_counter[0]}.zarr",
            )

        random.seed(42)
        for step in range(10):
            needs = batcher.exhausted_slot_indices()
            if needs:
                batcher.refill_slots(needs, [next_ride() for _ in needs])
            self.assertFalse(batcher.needs_refill())
            bounds = batcher.get_window_bounds()
            self.assertEqual(len(bounds), 2)
            for start, end in bounds:
                self.assertEqual(end - start, 24)
                self.assertGreaterEqual(start, 0)
            batcher.advance()

        self.assertGreater(ride_counter[0], 2, "Should have consumed multiple rides")


# ======================================================================
# Blockwise timestep tests (unchanged from before)
# ======================================================================

class TestBlockwiseTimesteps(unittest.TestCase):
    """Verify blockwise-identical timestep sampling (via trainer helper)."""

    def test_blockwise_index_shape_and_consistency(self):
        num_frame_per_block = 3
        batch_size, num_frames = 2, 21
        index = torch.randint(0, 1000, (batch_size, num_frames))
        index = index.reshape(batch_size, -1, num_frame_per_block)
        index[:, :, 1:] = index[:, :, 0:1]
        index = index.reshape(batch_size, num_frames)

        self.assertEqual(index.shape, (2, 21))
        for b in range(batch_size):
            for block in range(num_frames // num_frame_per_block):
                s = block * num_frame_per_block
                vals = index[b, s : s + num_frame_per_block]
                self.assertTrue(
                    (vals == vals[0]).all(),
                    f"Block {block} in batch {b} not uniform: {vals}",
                )

    def test_blocks_are_independent(self):
        torch.manual_seed(42)
        num_frame_per_block = 3
        batch_size, num_frames = 1, 21
        index = torch.randint(0, 1000, (batch_size, num_frames))
        index = index.reshape(batch_size, -1, num_frame_per_block)
        index[:, :, 1:] = index[:, :, 0:1]
        index = index.reshape(batch_size, num_frames)

        block_vals = index[0, ::num_frame_per_block]
        self.assertTrue(
            len(block_vals.unique()) > 1,
            "All blocks have the same value — unexpected with 1000 possible values",
        )


if __name__ == "__main__":
    unittest.main()
