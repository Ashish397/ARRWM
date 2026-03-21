"""Lockstep ride batcher for causal teacher training.

Provides ``LockstepRideBatcher`` — a deterministic non-overlapping
window batcher that advances multiple rides in lockstep.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class _RideSlot:
    """Bookkeeping for one batch slot's active ride."""

    zarr_path: str = ""
    prompt_embeds: Optional[torch.Tensor] = None
    z_actions: Optional[torch.Tensor] = None
    n_latent_frames: int = 0
    n_windows: int = 0


class LockstepRideBatcher:
    """Deterministic non-overlapping window batcher for multiple rides.

    Maintains ``batch_size`` slots.  All slots share the same window
    index and advance in lockstep.  When any slot's ride is exhausted,
    the entire group is replaced with the next ``batch_size`` rides.

    Parameters
    ----------
    window_size : int
        Latent frames per window (e.g. 21).
    num_frame_per_block : int
        Block size for block-aligned truncation (e.g. 3).
    batch_size : int
        Number of simultaneous rides.
    max_windows_per_ride : int or None
        Optional cap on how many windows to use from each ride.
    """

    def __init__(
        self,
        window_size: int = 21,
        num_frame_per_block: int = 3,
        batch_size: int = 1,
        max_windows_per_ride: Optional[int] = None,
    ):
        assert window_size % num_frame_per_block == 0
        self.window_size = window_size
        self.num_frame_per_block = num_frame_per_block
        self.batch_size = batch_size
        self.max_windows_per_ride = max_windows_per_ride

        self._slots: List[_RideSlot] = [_RideSlot() for _ in range(batch_size)]
        self._window_idx: int = 0
        self._group_n_windows: int = 0
        self._group_loaded: bool = False

    # ------------------------------------------------------------------
    # Group lifecycle
    # ------------------------------------------------------------------

    def load_group(self, ride_dicts: List[dict]) -> None:
        """Fill all slots with a new group of rides.

        Each element of *ride_dicts* must contain the keys returned by
        ``ZarrRideDataset.__getitem__``: ``zarr_path``, ``prompt_embeds``,
        ``z_actions``, ``n_latent_frames``.
        """
        if len(ride_dicts) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} rides, got {len(ride_dicts)}"
            )

        min_windows = None
        for i, rd in enumerate(ride_dicts):
            n_lat = int(rd["n_latent_frames"])
            n_win = n_lat // self.window_size
            if self.max_windows_per_ride is not None:
                n_win = min(n_win, self.max_windows_per_ride)
            self._slots[i] = _RideSlot(
                zarr_path=rd["zarr_path"],
                prompt_embeds=rd["prompt_embeds"],
                z_actions=rd["z_actions"],
                n_latent_frames=n_lat,
                n_windows=n_win,
            )
            if min_windows is None or n_win < min_windows:
                min_windows = n_win

        self._group_n_windows = min_windows or 0
        self._window_idx = 0
        self._group_loaded = True

    def needs_new_group(self) -> bool:
        """True when the current group is exhausted or not yet loaded."""
        if not self._group_loaded:
            return True
        return self._window_idx >= self._group_n_windows

    @property
    def current_window_idx(self) -> int:
        return self._window_idx

    @property
    def is_first_window(self) -> bool:
        return self._window_idx == 0

    @property
    def group_total_windows(self) -> int:
        return self._group_n_windows

    # ------------------------------------------------------------------
    # Per-step interface
    # ------------------------------------------------------------------

    def get_window_bounds(self) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` for each slot at the current window."""
        start = self._window_idx * self.window_size
        end = start + self.window_size
        return [(start, end)] * self.batch_size

    def get_slot_info(self) -> List[_RideSlot]:
        return list(self._slots)

    def advance(self) -> None:
        """Move to the next window."""
        self._window_idx += 1

    # ------------------------------------------------------------------
    # Convenience: build batched tensors
    # ------------------------------------------------------------------

    def load_latent_batch(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Lazy-load the current window's latents for all slots.

        Returns ``[B, window_size, C, H, W]``.
        """
        from utils.zarr_dataset import ZarrRideDataset

        bounds = self.get_window_bounds()
        chunks = []
        for slot, (start, end) in zip(self._slots, bounds):
            chunk = ZarrRideDataset.load_latent_chunk(
                slot.zarr_path, start, end,
            )
            chunks.append(chunk)
        return torch.stack(chunks).to(device=device, dtype=dtype)

    def load_z_actions_batch(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return z_actions for the current window, ``[B, window_size, D]``."""
        bounds = self.get_window_bounds()
        actions = []
        for slot, (start, end) in zip(self._slots, bounds):
            actions.append(slot.z_actions[start:end])
        return torch.stack(actions).to(device=device, dtype=dtype)

    def load_prompt_embeds_batch(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return prompt embeds for all slots, ``[B, L, C]``."""
        embeds = [slot.prompt_embeds for slot in self._slots]
        return torch.stack(embeds).to(device=device, dtype=dtype)

    def summary(self) -> str:
        """Human-readable one-line summary of current state."""
        names = [s.zarr_path.split("/")[-1] if s.zarr_path else "?" for s in self._slots]
        return (
            f"win={self._window_idx}/{self._group_n_windows} "
            f"rides=[{', '.join(names)}]"
        )
