#!/usr/bin/env python3
"""
Load the full FrodoBots dataset from dataset_cache.zarr + meta_data.

The full dataset (~164M steps, ~4.5k hours at 10 fps) lives on disk at a path
*outside* this repo, e.g.:
  - ~/fbots7k/extracted/frodobots_dataset
  - or set FRODOBOTS_DATASET_ROOT to that path.

train/*.arrow in that folder is only a small sample; use this loader to read
from dataset_cache.zarr with meta_data/episode_data_index.safetensors for
episode boundaries.
"""

from pathlib import Path
from typing import Optional, Union
import os

# Default path: same layout you have locally
def _default_root() -> Path:
    return Path(
        os.environ.get(
            "FRODOBOTS_DATASET_ROOT",
            str(Path.home() / "fbots7k" / "extracted" / "frodobots_dataset"),
        )
    ).resolve()


class FrodoBotsZarrDataset:
    """Read-only view of the full FrodoBots dataset (zarr + episode index)."""
    def __init__(self, root: Optional[Union[str, Path]] = None):
        self.root = Path(root or _default_root()).resolve()
        self._zarr = None
        self._from = None
        self._to = None
        self._load()
    def _load(self) -> None:
        import zarr
        from safetensors import safe_open
        zarr_path = self.root / "dataset_cache.zarr"
        idx_path = self.root / "meta_data" / "episode_data_index.safetensors"
        if not zarr_path.exists():
            raise FileNotFoundError(
                f"Zarr not found: {zarr_path}. Set FRODOBOTS_DATASET_ROOT or pass root=..."
            )
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Episode index not found: {idx_path}"
            )
        self._zarr = zarr.open(str(zarr_path), mode="r")
        with safe_open(str(idx_path), framework="pt") as sf:
            self._from = sf.get_tensor("from")
            self._to = sf.get_tensor("to")
    @property
    def zarr(self):
        """Raw zarr group (e.g. z['action'], z['episode_index'], z['observation.*'])."""
        return self._zarr

    @property
    def column_names(self):
        """Names of arrays/groups in the zarr (e.g. 'action', 'episode_index', 'observation.image')."""
        return list(self._zarr.keys())
    @property
    def num_episodes(self) -> int:
        return len(self._from)
    @property
    def total_steps(self) -> int:
        """Total steps in the zarr (may be larger than indexed range)."""
        return self._zarr["action"].shape[0]
    def episode_row_range(self, ep_idx: int):
        """Return (start, end) row indices for episode ep_idx (0-based)."""
        start = int(self._from[ep_idx].item())
        end = int(self._to[ep_idx].item())
        return start, end
    def get_episode_actions(
        self,
        ep_idx: int,
        start: int = 0,
        length: Optional[int] = None,
    ):
        """Load actions for one episode. Returns numpy array shape (N, 2).
        ep_idx: episode index (0-based).
        start: offset within the episode (default 0).
        length: number of steps (default: rest of episode).
        """
        import numpy as np
        s, e = self.episode_row_range(ep_idx)
        s = s + start
        if length is not None:
            e = min(e, s + length)
        return np.asarray(self._zarr["action"][s:e])
    def get_episode_slice(self, ep_idx: int, start: int = 0, length: Optional[int] = None):
        """Return (row_start, row_end) for slicing zarr arrays for this episode."""
        s, e = self.episode_row_range(ep_idx)
        s = s + start
        if length is not None:
            e = min(e, s + length)
        return s, e


def load_full_dataset(root: Optional[Union[str, Path]] = None) -> FrodoBotsZarrDataset:
    """Load the full FrodoBots dataset (zarr + episode index)."""
    return FrodoBotsZarrDataset(root=root)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else None
    ds = load_full_dataset(root)
    print("Root:", ds.root)
    print("Num episodes (indexed):", ds.num_episodes)
    print("Total steps (zarr):", ds.total_steps)
    print("Episode 0 row range:", ds.episode_row_range(0))
    print("First 5 actions of episode 0:\n", ds.get_episode_actions(0, length=5))
