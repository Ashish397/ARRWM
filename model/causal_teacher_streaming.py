"""Streaming ride batcher for causal teacher training.

Provides ``LockstepRideBatcher`` — a per-slot independent window batcher
that advances each batch slot through its own ride. When a slot's ride
is exhausted, only that slot is replaced with a fresh ride.

Each slot also receives a random block-aligned start offset so that
batch members begin at different temporal positions in their rides.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch


@dataclass
class _RideSlot:
    """Bookkeeping for one batch slot's active ride."""

    zarr_path: str = ""
    prompt_embeds: Optional[torch.Tensor] = None
    n_latent_frames: int = 0
    n_windows: int = 0
    window_idx: int = 0
    start_offset: int = 0
    loaded: bool = False


class LockstepRideBatcher:
    """Per-slot independent window batcher for multiple rides.

    Each batch slot tracks its own ride, window index, and start offset.
    When a slot's ride is exhausted the caller replaces just that slot
    via :meth:`refill_slots`.  The initial fill is done through
    :meth:`load_group` for backward compatibility.

    Parameters
    ----------
    window_size : int
        Latent frames per window (e.g. 21).
    num_frame_per_block : int
        Block size for block-aligned start offsets (e.g. 3).
    batch_size : int
        Number of simultaneous rides.
    max_windows_per_ride : int or None
        Optional cap on how many windows to use from each ride.
    context_frames : int
        Clean context frames prepended to each window.
    max_start_offset : int
        Upper bound (inclusive) for per-slot random start offset.
        Actual offset is rounded down to a multiple of
        ``num_frame_per_block``.
    """

    def __init__(
        self,
        window_size: int = 21,
        num_frame_per_block: int = 3,
        batch_size: int = 1,
        max_windows_per_ride: Optional[int] = None,
        context_frames: int = 3,
        max_start_offset: int = 40,
    ):
        assert window_size % num_frame_per_block == 0
        self.window_size = window_size
        self.num_frame_per_block = num_frame_per_block
        self.batch_size = batch_size
        self.max_windows_per_ride = max_windows_per_ride
        self.context_frames = context_frames
        self.max_start_offset = max_start_offset

        self._slots: List[_RideSlot] = [_RideSlot() for _ in range(batch_size)]

    # ------------------------------------------------------------------
    # Slot-level helpers
    # ------------------------------------------------------------------

    def _random_start_offset(self) -> int:
        """Sample a block-aligned start offset in ``[0, max_start_offset]``."""
        if self.max_start_offset <= 0:
            return 0
        raw = random.randint(0, self.max_start_offset)
        return (raw // self.num_frame_per_block) * self.num_frame_per_block

    def _init_slot(self, idx: int, rd: dict) -> None:
        """Populate slot *idx* from a ride dict with a fresh random offset."""
        n_lat = int(rd["n_latent_frames"])
        offset = self._random_start_offset()
        usable = n_lat - offset - self.context_frames
        if usable < self.window_size:
            offset = 0
            usable = n_lat - self.context_frames
        n_win = max(0, usable // self.window_size)
        if self.max_windows_per_ride is not None:
            n_win = min(n_win, self.max_windows_per_ride)
        self._slots[idx] = _RideSlot(
            zarr_path=rd["zarr_path"],
            prompt_embeds=rd["prompt_embeds"],
            n_latent_frames=n_lat,
            n_windows=n_win,
            window_idx=0,
            start_offset=offset,
            loaded=True,
        )

    # ------------------------------------------------------------------
    # Group lifecycle (backward-compatible initial fill)
    # ------------------------------------------------------------------

    def load_group(self, ride_dicts: List[dict]) -> None:
        """Fill all slots with a new group of rides.

        Each element of *ride_dicts* must contain ``zarr_path``,
        ``prompt_embeds``, ``n_latent_frames``.
        """
        if len(ride_dicts) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} rides, got {len(ride_dicts)}"
            )
        for i, rd in enumerate(ride_dicts):
            self._init_slot(i, rd)

    # ------------------------------------------------------------------
    # Per-slot refill
    # ------------------------------------------------------------------

    def exhausted_slot_indices(self) -> List[int]:
        """Return indices of slots that need a new ride."""
        return [
            i for i, s in enumerate(self._slots)
            if not s.loaded or s.window_idx >= s.n_windows
        ]

    def refill_slots(self, slot_indices: List[int], ride_dicts: List[dict]) -> None:
        """Replace specific exhausted slots with fresh rides."""
        if len(slot_indices) != len(ride_dicts):
            raise ValueError(
                f"Got {len(slot_indices)} indices but {len(ride_dicts)} rides"
            )
        for idx, rd in zip(slot_indices, ride_dicts):
            self._init_slot(idx, rd)

    def needs_refill(self) -> bool:
        """True if any slot needs a new ride, or batch not yet initialised."""
        return len(self.exhausted_slot_indices()) > 0

    def needs_new_group(self) -> bool:
        """Backward-compatible alias for :meth:`needs_refill`."""
        return self.needs_refill()

    # ------------------------------------------------------------------
    # Properties (backward-compatible)
    # ------------------------------------------------------------------

    @property
    def current_window_idx(self) -> int:
        """Max window index across all active slots (for logging)."""
        active = [s.window_idx for s in self._slots if s.loaded]
        return max(active) if active else 0

    @property
    def is_first_window(self) -> bool:
        return all(s.window_idx == 0 for s in self._slots if s.loaded)

    @property
    def group_total_windows(self) -> int:
        """Minimum n_windows across all active slots (for logging)."""
        active = [s.n_windows for s in self._slots if s.loaded]
        return min(active) if active else 0

    # ------------------------------------------------------------------
    # Per-step interface
    # ------------------------------------------------------------------

    def get_window_bounds(self) -> List[Tuple[int, int]]:
        """Return ``[(start, end), ...]`` for each slot at its current window.

        Each slot has its own start offset and window index, so bounds
        may differ across slots.  When ``context_frames > 0`` the
        returned range includes the leading context.
        """
        bounds = []
        for s in self._slots:
            start = s.start_offset + s.window_idx * self.window_size
            end = start + self.window_size + self.context_frames
            bounds.append((start, end))
        return bounds

    def get_slot_info(self) -> List[_RideSlot]:
        return list(self._slots)

    def advance(self) -> None:
        """Advance every non-exhausted slot to the next window."""
        for s in self._slots:
            if s.loaded and s.window_idx < s.n_windows:
                s.window_idx += 1

    # ------------------------------------------------------------------
    # Convenience: build batched tensors
    # ------------------------------------------------------------------

    def load_latent_batch(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Lazy-load the current window's latents for all slots.

        Returns ``[B, window_size + context_frames, C, H, W]``.
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
        encode_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Return z_actions for the current window, ``[B, window_size + context_frames, D]``.

        Parameters
        ----------
        encode_fn : callable
            ``(zarr_path, n_latent_frames, start, end) -> Tensor[end-start, D]``
            Encodes motion on-the-fly for just the current window.
        """
        if encode_fn is None:
            raise RuntimeError(
                "load_z_actions_batch requires encode_fn; "
                "per-ride z_actions are no longer pre-computed."
            )
        bounds = self.get_window_bounds()
        actions = []
        for slot, (start, end) in zip(self._slots, bounds):
            actions.append(
                encode_fn(slot.zarr_path, slot.n_latent_frames, start, end)
            )
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
        parts = []
        for s in self._slots:
            name = s.zarr_path.split("/")[-1] if s.zarr_path else "?"
            parts.append(f"{name}[w={s.window_idx}/{s.n_windows},off={s.start_offset}]")
        return f"slots=[{', '.join(parts)}]"
