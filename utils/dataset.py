# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from torch.utils.data import Dataset
import torch
import lmdb
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import os
import random
from typing import Callable, Optional
try:
    import datasets
except ImportError:
    datasets = None


STYLE_SUFFIX = ('Captured with a low-mounted wide-angle dash/action camera (around 100 degrees HFOV and 70 degrees VFOV) using fixed focus and auto-exposure, '                 'yielding mild fisheye distortion, soft corners, faint vignette, and minimal stabilization at 480p. Small-sensor look with occasional '                 'starburst/ghosting on point lights and occational smudges on the dome lens. The video is taken in the real world.')


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        if datasets is None:
            raise ImportError("The 'datasets' package is required for MultiTextDataset")
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


class VideoLatentCaptionDataset(Dataset):
    """Dataset pairing pre-encoded video latents with first-chunk captions."""

    @dataclass(frozen=True)
    class Sample:
        latent_path: Path
        encoded_path: Optional[Path] = None
        prompt_text: Optional[str] = None
        actions_path: Optional[Path] = None

    def __init__(
        self,
        latent_root: str,
        caption_root: str,
        num_frames: int = 21,
        *,
        text_pre_encoded: bool = False,
        encoded_suffix: str = "_encoded",
        action_root: Optional[str] = None,
        include_dir_substrings: Optional[list[str]] = None,
        include_actions: bool = True,
        blacklist_path: Optional[str] = None,
        undersample_actions: bool = True,
    ):
        self.latent_root = Path(latent_root)
        self.caption_root = Path(caption_root)
        self.num_frames = num_frames // 3
        self.text_pre_encoded = text_pre_encoded
        self.encoded_suffix = encoded_suffix
        self.include_actions = include_actions
        self.undersample_actions = undersample_actions

        if not self.latent_root.exists():
            raise FileNotFoundError(f"Latent root does not exist: {latent_root}")
        if not self.caption_root.exists():
            raise FileNotFoundError(f"Caption root does not exist: {caption_root}")

        self.actions_root = self._resolve_action_root(action_root) if include_actions else None
        self._action_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
        self._blacklisted_dirs: set[Path] = set()
        
        if blacklist_path:
            self._load_blacklist(Path(blacklist_path))

        # Index all files
        latent_map = self._index_latents()
        encoded_map = self._index_files(self.caption_root, f"*{encoded_suffix}.json")
        raw_map = self._index_files(self.caption_root, "*.json", skip=lambda p: p.name.endswith(f"{encoded_suffix}.json"))
        actions_map = self._index_files(self.actions_root, "input_actions_*.csv") if include_actions else {}

        # Apply filters
        if include_dir_substrings:
            filters = tuple(include_dir_substrings)
            latent_map = {k: v for k, v in latent_map.items() if any(s in str(k) for s in filters)}
            encoded_map = {k: v for k, v in encoded_map.items() if any(s in str(k) for s in filters)}
            raw_map = {k: v for k, v in raw_map.items() if any(s in str(k) for s in filters)}
            actions_map = {k: v for k, v in actions_map.items() if any(s in str(k) for s in filters)}

        # Build samples
        self.samples: list[VideoLatentCaptionDataset.Sample] = []
        all_dirs = sorted(set(latent_map) | set(encoded_map) | set(raw_map) | set(actions_map))
        
        for rel_dir in all_dirs:
            if rel_dir in self._blacklisted_dirs:
                continue
            latent_paths = latent_map.get(rel_dir, [])
            if not latent_paths:
                continue

            # Build sample (handles both encoded and raw captions)
            sample = self._build_sample(rel_dir, latent_paths[0], encoded_map, raw_map, actions_map)
            if sample is None:
                continue

            self.samples.append(sample)
            # Add additional latent files from same directory
            for latent_path in latent_paths[1:]:
                self.samples.append(self.Sample(
                    latent_path=latent_path,
                    encoded_path=sample.encoded_path,
                    prompt_text=sample.prompt_text,
                    actions_path=sample.actions_path,
                ))

        if not self.samples:
            raise RuntimeError("No paired latent/caption samples were found.")

    def _resolve_action_root(self, action_root: Optional[str]) -> Path:
        if action_root:
            candidate = Path(action_root)
        else:
            parent = self.latent_root.parent
            if parent.name.endswith("_encoded"):
                base_name = parent.name[:-len("_encoded")] + "_actions"
                candidate = parent.with_name(base_name) / self.latent_root.name
            else:
                candidate = parent / self.latent_root.name
        if not candidate.exists():
            raise FileNotFoundError(f"Action root does not exist: {candidate}")
        return candidate

    def _index_latents(self) -> dict[Path, list[Path]]:
        """Index all latent video files."""
        latents: dict[Path, list[Path]] = {}
        for path in sorted(self.latent_root.rglob("encoded_video_*.pt")):
            if path.is_file():
                try:
                    rel_dir = path.parent.relative_to(self.latent_root)
                    latents.setdefault(rel_dir, []).append(path)
                except ValueError:
                    continue
        return latents

    def _index_files(self, root: Optional[Path], pattern: str, *, skip: Optional[Callable[[Path], bool]] = None) -> dict[Path, Path]:
        """Index files matching pattern, returning dict of rel_dir -> file_path."""
        if root is None:
            return {}
        matches: dict[Path, Path] = {}
        for path in sorted(root.rglob(pattern)):
            if not path.is_file() or (skip and skip(path)):
                continue
            try:
                rel_dir = path.parent.relative_to(root)
                if rel_dir in matches and matches[rel_dir] != path:
                    raise RuntimeError(f"Multiple {pattern} files found for {rel_dir}")
                matches[rel_dir] = path
            except ValueError:
                continue
        return matches

    def _load_blacklist(self, blacklist_file: Path) -> None:
        """Load blacklisted directories from file."""
        if not blacklist_file.is_file():
            raise FileNotFoundError(f"Blacklist file not found: {blacklist_file}")
        
        with blacklist_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                
                entry = line.split(":", 1)[0].strip()
                if entry:
                    path = Path(entry)
                    try:
                        rel_path = path.relative_to(self.latent_root)
                        if rel_path.suffix == ".pt":
                            rel_path = rel_path.parent
                        self._blacklisted_dirs.add(rel_path)
                    except ValueError:
                        self._blacklisted_dirs.add(path)

    def _build_sample(
        self,
        rel_dir: Path,
        latent_path: Path,
        encoded_map: dict[Path, Path],
        raw_map: dict[Path, Path],
        actions_map: dict[Path, Path],
    ) -> Optional[Sample]:
        """Build a sample, preferring encoded captions but falling back to raw."""
        actions_path = actions_map.get(rel_dir) if self.include_actions else None
        if self.include_actions and actions_path is None:
            return None

        # Prefer encoded caption
        encoded_path = encoded_map.get(rel_dir)
        if encoded_path:
            return self.Sample(latent_path=latent_path, encoded_path=encoded_path, actions_path=actions_path)

        # Fall back to raw caption
        raw_path = raw_map.get(rel_dir)
        if raw_path is None:
            return None

        prompt_text = self._load_prompt(raw_path)
        if prompt_text is None:
            return None

        return self.Sample(latent_path=latent_path, prompt_text=prompt_text, actions_path=actions_path)

    def _load_prompt(self, caption_path: Path) -> Optional[str]:
        """Load prompt text from JSON caption file."""
        try:
            with caption_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            return None

        caption = (data.get("combined_analysis") or "").strip()
        if not caption:
            return None

        return f"{caption} {STYLE_SUFFIX}"

    def _load_actions(self, actions_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Load actions from CSV file, returning (frame_ids, values) tensors."""
        if actions_path in self._action_cache:
            return self._action_cache[actions_path]

        try:
            with actions_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None or "frame_id" not in reader.fieldnames:
                    raise ValueError("Missing 'frame_id' column")
                value_keys = [k for k in reader.fieldnames if k != "frame_id"]
                if not value_keys:
                    raise ValueError("No action columns found")

                frames, values = [], []
                for row in reader:
                    frames.append(int(row["frame_id"]))
                    values.append([float(row[k]) for k in value_keys])
        except Exception as exc:
            raise RuntimeError(f"Failed to read actions from {actions_path}: {exc}") from exc

        if not frames:
            raise ValueError(f"Actions file {actions_path} is empty")

        frame_tensor = torch.tensor(frames, dtype=torch.long)
        value_tensor = torch.tensor(values, dtype=torch.float32)
        self._action_cache[actions_path] = (frame_tensor, value_tensor)
        return frame_tensor, value_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Get a sample, handling frame windowing and NaN filtering."""
        sample = self.samples[idx % len(self.samples)]
        
        # Load latent
        latents = torch.load(sample.latent_path, map_location="cpu")[0]
        if not isinstance(latents, torch.Tensor) or latents.ndim != 4 or latents.shape[0] == 0:
            raise ValueError(f"Invalid latent format at {sample.latent_path}")

        #reshape latents such that
        latents = latents[1:]
        latents = latents[:(latents.shape[0]//3)*3].unsqueeze(0)
        latents = latents.reshape(latents.shape[1]//3, 3, 16, 60, 104)

        # Load and sync actions if needed
        if self.include_actions:
            if sample.actions_path is None:
                raise RuntimeError("Actions path missing for sample")
            
            _, action_values = self._load_actions(sample.actions_path)
            
            # Sync: find minimum length and trim all to that
            min_length = min(latents.shape[0], action_values.shape[0])
            latents = latents[:min_length]
            action_values = action_values[:min_length]

            # Filter NaN/inf (inspired by train_motion_2_action.py)
            latents_valid = ~(torch.isnan(latents).any(dim=(1, 2, 3, 4)) | torch.isinf(latents).any(dim=(1, 2, 3, 4)))
            actions_valid = torch.isfinite(action_values).all(dim=1)
            valid_mask = latents_valid & actions_valid

            if valid_mask.sum() < self.num_frames:
                # Retry with next sample if not enough valid frames
                return self.__getitem__(idx + 1)

            # Apply mask
            latents = latents[valid_mask]
            action_values = action_values[valid_mask]

            # Undersample popular action categories (linear dominant and noop)
            if self.undersample_actions and action_values.shape[1] >= 2:
                # Only keep 10% of linear dominant samples
                # Check linear dominance: first action dimension dominates second and > 0.1
                ld = (torch.abs(action_values[:, 0]) > torch.abs(action_values[:, 1])) & (action_values[:, 0] > 0.1)
                # No-op: both action dimensions are < 0.1
                noop = (torch.abs(action_values[:, 0]) < 0.1) & (torch.abs(action_values[:, 1]) < 0.1)
                # Random sampling: keep 10% of linear dominant, 4% of noop, 100% of others
                r = torch.rand(action_values.shape[0], device=action_values.device)
                keep = (~ld & ~noop) | (ld & (r < 0.1)) | (noop & (r < 0.2))
                
                if keep.sum() < self.num_frames:
                    # Retry with next sample if not enough frames remain after undersampling
                    return self.__getitem__(idx + 1)
                
                # Apply undersampling mask
                latents = latents[keep]
                action_values = action_values[keep]

            # Select random window
            max_start = len(latents) - self.num_frames
            start = random.randint(1, max_start) if max_start > 0 else 0
            end = start + self.num_frames

            latents_slice = latents[start:end].contiguous().float().reshape(self.num_frames*3, 16, 60, 104)
            #repeat the elements of action values 3 times such that [1,2,3] -> [1,1,1,2,2,2,3,3,3]
            action_values_slice = action_values[start:end].repeat_interleave(3, dim=0)

            sample_dict = {
                "idx": idx,
                "prompts": sample.prompt_text or "",
                "real_latents": latents_slice,
                "actions": action_values_slice.clone(),
            }
        else:
            # No actions: simple random window
            total = latents.shape[0]
            if total < self.num_frames:
                raise RuntimeError(f"Not enough frames in sample: {total} < {self.num_frames}")
            
            max_start = total - self.num_frames
            start = random.randint(0, max_start) if max_start > 0 else 0
            end = start + self.num_frames

            latents_slice = latents[start:end].contiguous().float().reshape(self.num_frames*3, 16, 60, 104)
            sample_dict = {
                "idx": idx,
                "prompts": sample.prompt_text or "",
                "real_latents": latents_slice,
            }

        # Load encoded caption if available
        if sample.encoded_path:
            try:
                with sample.encoded_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                embedding = payload.get("caption_encoded")
                if embedding is None:
                    raise ValueError(f"'caption_encoded' missing in {sample.encoded_path}")
                prompt_embeds = torch.tensor(embedding, dtype=torch.float32)
                if prompt_embeds.ndim != 2:
                    raise ValueError(f"Encoded caption has unexpected shape {prompt_embeds.shape}")
                sample_dict["prompt_embeds"] = prompt_embeds
            except Exception as exc:
                raise RuntimeError(f"Failed to read encoded caption from {sample.encoded_path}: {exc}") from exc
        elif self.text_pre_encoded and not sample.prompt_text:
            raise RuntimeError("Encoded caption path missing for sample with pre-encoded text.")

        return sample_dict


def cycle(dl):
    while True:
        for data in dl:
            yield data
