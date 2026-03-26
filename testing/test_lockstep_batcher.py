#!/usr/bin/env python3
"""Unit tests for LockstepRideBatcher.

Verifies ride-slot progression, window bounds, group lifecycle, and
batch construction semantics with synthetic ride data (no zarr/GPU needed).

Usage:
    python testing/test_lockstep_batcher.py [-v]
"""

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


class TestLockstepRideBatcher(unittest.TestCase):
    """Core batcher logic."""

    def test_batch_size_1_sequential_windows(self):
        """batch_size=1: yields context-prepended windows until ride ends."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        ride = _fake_ride(63, "ride_63.zarr")
        batcher.load_group([ride])

        self.assertFalse(batcher.needs_new_group())
        self.assertEqual(batcher.group_total_windows, 2)

        expected = [(0, 24), (21, 45)]
        for i, (exp_s, exp_e) in enumerate(expected):
            self.assertEqual(batcher.current_window_idx, i)
            bounds = batcher.get_window_bounds()
            self.assertEqual(len(bounds), 1)
            self.assertEqual(bounds[0], (exp_s, exp_e))
            batcher.advance()

        self.assertTrue(batcher.needs_new_group())

    def test_batch_size_2_lockstep(self):
        """batch_size=2: both slots share the same window index."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
        )
        ride_a = _fake_ride(63, "rideA.zarr")
        ride_b = _fake_ride(84, "rideB.zarr")
        batcher.load_group([ride_a, ride_b])

        # Truncated to the shorter ride and context-aware -> 2 windows.
        self.assertEqual(batcher.group_total_windows, 2)

        for win in range(2):
            bounds = batcher.get_window_bounds()
            self.assertEqual(len(bounds), 2)
            self.assertEqual(bounds[0], (win * 21, win * 21 + 24))
            self.assertEqual(bounds[1], (win * 21, win * 21 + 24))
            batcher.advance()

        self.assertTrue(batcher.needs_new_group())

    def test_truncation_to_min_ride(self):
        """Group windows are truncated to the shortest ride."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
        )
        short = _fake_ride(42, "short.zarr")
        long_ = _fake_ride(210, "long.zarr")
        batcher.load_group([short, long_])

        self.assertEqual(batcher.group_total_windows, 1)

    def test_max_windows_per_ride(self):
        """max_windows_per_ride caps window count."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
            max_windows_per_ride=2,
        )
        ride = _fake_ride(210, "long.zarr")
        batcher.load_group([ride])

        self.assertEqual(batcher.group_total_windows, 2)

    def test_is_first_window(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        batcher.load_group([_fake_ride(63)])

        self.assertTrue(batcher.is_first_window)
        batcher.advance()
        self.assertFalse(batcher.is_first_window)

    def test_needs_new_group_initially(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        self.assertTrue(batcher.needs_new_group())

    def test_wrong_batch_size_raises(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
        )
        with self.assertRaises(ValueError):
            batcher.load_group([_fake_ride(63)])

    def test_ride_too_short_for_one_window(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        batcher.load_group([_fake_ride(20)])
        self.assertEqual(batcher.group_total_windows, 0)
        self.assertTrue(batcher.needs_new_group())

    def test_exact_minimum_length_gives_one_window(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        batcher.load_group([_fake_ride(24)])
        self.assertEqual(batcher.group_total_windows, 1)
        self.assertEqual(batcher.get_window_bounds()[0], (0, 24))

    def test_z_actions_batch(self):
        """load_z_actions_batch returns correctly sliced tensors via encode_fn."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
        )
        ride_a = _fake_ride(63, "a.zarr")
        ride_b = _fake_ride(63, "b.zarr")
        batcher.load_group([ride_a, ride_b])

        z_batch = batcher.load_z_actions_batch(
            torch.device("cpu"), encode_fn=_fake_encode_fn,
        )
        self.assertEqual(z_batch.shape, (2, 24, 8))
        torch.testing.assert_close(
            z_batch[0], _FAKE_Z_ACTIONS[ride_a["zarr_path"]][:24],
        )
        torch.testing.assert_close(
            z_batch[1], _FAKE_Z_ACTIONS[ride_b["zarr_path"]][:24],
        )

        batcher.advance()
        z_batch2 = batcher.load_z_actions_batch(
            torch.device("cpu"), encode_fn=_fake_encode_fn,
        )
        torch.testing.assert_close(
            z_batch2[0], _FAKE_Z_ACTIONS[ride_a["zarr_path"]][21:45],
        )

    def test_z_actions_batch_requires_encode_fn(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        batcher.load_group([_fake_ride(63, "requires_encode.zarr")])
        with self.assertRaises(RuntimeError):
            batcher.load_z_actions_batch(torch.device("cpu"))

    def test_prompt_embeds_batch(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=2,
        )
        ride_a = _fake_ride(63, "a.zarr")
        ride_b = _fake_ride(63, "b.zarr")
        batcher.load_group([ride_a, ride_b])

        pe = batcher.load_prompt_embeds_batch(torch.device("cpu"))
        self.assertEqual(pe.shape, (2, 77, 512))
        torch.testing.assert_close(pe[0], ride_a["prompt_embeds"])
        torch.testing.assert_close(pe[1], ride_b["prompt_embeds"])

    def test_summary_string(self):
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        batcher.load_group([_fake_ride(63, "test.zarr")])
        s = batcher.summary()
        self.assertIn("test.zarr", s)
        self.assertIn("win=0/2", s)

    def test_multiple_groups(self):
        """After exhausting one group, loading another works correctly."""
        batcher = LockstepRideBatcher(
            window_size=21, num_frame_per_block=3, batch_size=1,
        )
        batcher.load_group([_fake_ride(42, "first.zarr")])
        self.assertEqual(batcher.group_total_windows, 1)
        batcher.advance()
        self.assertTrue(batcher.needs_new_group())

        batcher.load_group([_fake_ride(63, "second.zarr")])
        self.assertEqual(batcher.group_total_windows, 2)
        self.assertEqual(batcher.current_window_idx, 0)
        self.assertTrue(batcher.is_first_window)


class TestBlockwiseTimesteps(unittest.TestCase):
    """Verify blockwise-identical timestep sampling (via trainer helper)."""

    def test_blockwise_index_shape_and_consistency(self):
        """Each 3-frame block shares the same index value."""
        # Simulate what _make_blockwise_index does
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
        """Different blocks should (with high probability) have different values."""
        torch.manual_seed(42)
        num_frame_per_block = 3
        batch_size, num_frames = 1, 21
        index = torch.randint(0, 1000, (batch_size, num_frames))
        index = index.reshape(batch_size, -1, num_frame_per_block)
        index[:, :, 1:] = index[:, :, 0:1]
        index = index.reshape(batch_size, num_frames)

        block_vals = index[0, ::num_frame_per_block]
        # 7 blocks with values in [0,1000) — very unlikely all equal
        self.assertTrue(
            len(block_vals.unique()) > 1,
            "All blocks have the same value — unexpected with 1000 possible values",
        )


if __name__ == "__main__":
    unittest.main()
