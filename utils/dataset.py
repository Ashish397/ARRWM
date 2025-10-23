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
        action_variant: str = "input",
    ):
        self.latent_root = Path(latent_root)
        self.caption_root = Path(caption_root)
        self.num_frames = num_frames
        self.text_pre_encoded = text_pre_encoded
        self.encoded_suffix = encoded_suffix

        if not self.latent_root.exists():
            raise FileNotFoundError(f"Latent root does not exist: {latent_root}")
        if not self.caption_root.exists():
            raise FileNotFoundError(f"Caption root does not exist: {caption_root}")

        if action_variant not in {"input", "output"}:
            raise ValueError(
                f"Unsupported action_variant '{action_variant}' (expected 'input' or 'output')."
            )
        self.action_variant = action_variant
        self.actions_root = self._resolve_action_root(action_root)

        self._action_cache: dict[Path, tuple[torch.Tensor, torch.Tensor]] = {}
        self._action_value_keys: Optional[tuple[str, ...]] = None
        self._short_action_warned: set[Path] = set()

        latent_map = self._index_latents(self.latent_root)
        encoded_map = self._index_unique(
            self.caption_root,
            f"*{self.encoded_suffix}.json",
            "encoded captions",
        )
        raw_map = self._index_unique(
            self.caption_root,
            "*.json",
            "captions",
            skip=lambda p: p.name.endswith(f"{self.encoded_suffix}.json"),
        )
        actions_map = self._index_unique(
            self.actions_root,
            f"{self.action_variant}_actions_*.csv",
            f"{self.action_variant} actions",
        )

        self.samples: list["VideoLatentCaptionDataset.Sample"] = []
        all_dirs = sorted(
            set(latent_map) | set(encoded_map) | set(raw_map) | set(actions_map)
        )
        for rel_dir in all_dirs:
            latent_paths = latent_map.get(rel_dir, [])
            if not latent_paths:
                print(f"{rel_dir}: video doesn't exist, skipping")
                continue

            first_latent = latent_paths[0]
            if self.text_pre_encoded:
                base_sample = self._build_encoded_sample(
                    rel_dir,
                    first_latent,
                    encoded_map,
                    raw_map,
                    actions_map,
                )
            else:
                base_sample = self._build_raw_sample(
                    rel_dir,
                    first_latent,
                    raw_map,
                    actions_map,
                )

            if base_sample is None:
                continue

            self.samples.append(base_sample)
            for latent_path in latent_paths[1:]:
                self.samples.append(
                    self.Sample(
                        latent_path=latent_path,
                        encoded_path=base_sample.encoded_path,
                        prompt_text=base_sample.prompt_text,
                        actions_path=base_sample.actions_path,
                    )
                )

        if not self.samples:
            raise RuntimeError("No paired latent/caption samples were found.")

    def _resolve_action_root(self, action_root: Optional[str]) -> Path:
        if action_root is not None:
            candidate = Path(action_root)
        else:
            parent = self.latent_root.parent
            if parent.name.endswith("_encoded"):
                base_name = parent.name[: -len("_encoded")] + "_actions"
                candidate = parent.with_name(base_name) / self.latent_root.name
            else:
                candidate = parent / self.latent_root.name
        if not candidate.exists():
            raise FileNotFoundError(f"Action root does not exist: {candidate}")
        return candidate

    def _index_latents(self, root: Path) -> dict[Path, list[Path]]:
        latents: dict[Path, list[Path]] = {}
        for path in sorted(root.rglob("encoded_video_*.pt")):
            if not path.is_file():
                continue
            try:
                rel_dir = path.parent.relative_to(root)
            except ValueError:
                continue
            latents.setdefault(rel_dir, []).append(path)
        return latents

    def _index_unique(
        self,
        root: Path,
        pattern: str,
        description: str,
        *,
        skip: Optional[Callable[[Path], bool]] = None,
    ) -> dict[Path, Path]:
        matches: dict[Path, Path] = {}
        for path in sorted(root.rglob(pattern)):
            if not path.is_file():
                continue
            if skip and skip(path):
                continue
            try:
                rel_dir = path.parent.relative_to(root)
            except ValueError:
                continue
            prev = matches.get(rel_dir)
            if prev is not None and prev != path:
                raise RuntimeError(f"Multiple {description} found for {rel_dir}")
            matches[rel_dir] = path
        return matches

    def _build_encoded_sample(
        self,
        rel_dir: Path,
        latent_path: Path,
        encoded_map: dict[Path, Path],
        raw_map: dict[Path, Path],
        actions_map: dict[Path, Path],
    ) -> Optional["VideoLatentCaptionDataset.Sample"]:
        actions_path = actions_map.get(rel_dir)
        if actions_path is None:
            print(f"{rel_dir}: actions don't exist, skipping")
            return None

        encoded_path = encoded_map.get(rel_dir)
        if encoded_path is not None:
            return self.Sample(
                latent_path=latent_path,
                encoded_path=encoded_path,
                actions_path=actions_path,
            )

        raw_path = raw_map.get(rel_dir)
        if raw_path is None:
            print(f"{rel_dir}: encoded caption doesn't exist, skipping")
            return None

        prompt_text = self._load_prompt(raw_path)
        if prompt_text is None:
            return None

        print(f"{rel_dir}: encoded caption missing, falling back to raw caption")
        return self.Sample(
            latent_path=latent_path,
            prompt_text=prompt_text,
            actions_path=actions_path,
        )

    def _build_raw_sample(
        self,
        rel_dir: Path,
        latent_path: Path,
        raw_map: dict[Path, Path],
        actions_map: dict[Path, Path],
    ) -> Optional["VideoLatentCaptionDataset.Sample"]:
        actions_path = actions_map.get(rel_dir)
        if actions_path is None:
            print(f"{rel_dir}: actions don't exist, skipping")
            return None

        raw_path = raw_map.get(rel_dir)
        if raw_path is None:
            print(f"{rel_dir}: caption doesn't exist, skipping")
            return None

        prompt_text = self._load_prompt(raw_path)
        if prompt_text is None:
            return None

        return self.Sample(
            latent_path=latent_path,
            prompt_text=prompt_text,
            actions_path=actions_path,
        )

    def _load_prompt(self, caption_path: Path) -> Optional[str]:
        try:
            with caption_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"{caption_path}: failed to load caption ({exc}), skipping")
            return None

        caption = (data.get("combined_analysis") or "").strip()
        if not caption:
            print(f"{caption_path}: combined_analysis missing, skipping")
            return None

        return f"{caption} {STYLE_SUFFIX}"

    def _load_actions_tensor(self, actions_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._action_cache.get(actions_path)
        if cached is not None:
            return cached

        try:
            with actions_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None or "frame_id" not in reader.fieldnames:
                    raise ValueError("Missing 'frame_id' column in actions file")
                value_keys = [name for name in reader.fieldnames if name != "frame_id"]
                if not value_keys:
                    raise ValueError("No action columns found in actions file")
                frames = []
                values = []
                for row in reader:
                    frames.append(int(row["frame_id"]))
                    values.append([float(row[key]) for key in value_keys])
        except Exception as exc:
            raise RuntimeError(f"Failed to read actions from {actions_path}: {exc}") from exc

        if not frames:
            raise ValueError(f"Actions file {actions_path} is empty")

        frame_tensor = torch.tensor(frames, dtype=torch.long)
        value_tensor = torch.tensor(values, dtype=torch.float32)

        if self._action_value_keys is None:
            self._action_value_keys = tuple(value_keys)
        elif tuple(value_keys) != self._action_value_keys:
            raise RuntimeError(
                f"Action columns mismatch in {actions_path}; expected {self._action_value_keys}, got {tuple(value_keys)}"
            )

        self._action_cache[actions_path] = (frame_tensor, value_tensor)
        return frame_tensor, value_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        latents = torch.load(sample.latent_path, map_location="cpu")[0]
        if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
            raise ValueError(f"Unexpected latent format at {sample.latent_path}")
        if latents.shape[0] == 0:
            raise ValueError(f"Latent sequence empty: {sample.latent_path}")

        total = latents.shape[0]
        repeat = 0
        if total >= self.num_frames:
            start_max = total - self.num_frames
            start = random.randint(0, start_max) if start_max > 0 else 0
            end = start + self.num_frames
        else:
            start = 0
            end = total
            repeat = self.num_frames - total

        latents = latents[start:end]
        if repeat:
            latents = torch.cat(
                [latents, latents[-1:].repeat(repeat, 1, 1, 1)],
                dim=0,
            )

        latents = latents.contiguous().float()
        prompt_text = sample.prompt_text or ""
        sample_dict = {
            "idx": idx,
            "prompts": prompt_text,
            "real_latents": latents,
        }

        actions_path = sample.actions_path
        if actions_path is None:
            raise RuntimeError("Actions path missing for sample.")

        action_frames, action_values = self._load_actions_tensor(actions_path)
        available = action_values.shape[0]
        desired = end - start
        shortfall = 0
        if start >= available:
            shortfall = desired
            base_frame = action_frames[-1:]
            base_value = action_values[-1:]
            action_frames = base_frame.repeat(desired)
            action_values = base_value.repeat(desired, 1)
        else:
            slice_end = min(end, available)
            action_frames = action_frames[start:slice_end]
            action_values = action_values[start:slice_end]
            shortfall = desired - action_values.shape[0]
            if shortfall > 0:
                pad_frames = action_frames[-1:].repeat(shortfall)
                pad_values = action_values[-1:].repeat(shortfall, 1)
                action_frames = torch.cat([action_frames, pad_frames], dim=0)
                action_values = torch.cat([action_values, pad_values], dim=0)

        if shortfall > 0 and actions_path not in self._short_action_warned:
            print(
                f"{actions_path}: actions shorter than video, repeating last action for {shortfall} frames"
            )
            self._short_action_warned.add(actions_path)

        if repeat:
            action_frames = torch.cat(
                [action_frames, action_frames[-1:].repeat(repeat)],
                dim=0,
            )
            action_values = torch.cat(
                [action_values, action_values[-1:].repeat(repeat, 1)],
                dim=0,
            )

        sample_dict["action_frames"] = action_frames.clone()
        sample_dict["actions"] = action_values.clone()

        if sample.encoded_path is not None:
            try:
                with sample.encoded_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read encoded caption from {sample.encoded_path}: {exc}"
                ) from exc

            embedding = payload.get("caption_encoded")
            if embedding is None:
                raise ValueError(f"'caption_encoded' missing in {sample.encoded_path}")

            prompt_embeds = torch.tensor(embedding, dtype=torch.float32)
            if prompt_embeds.ndim != 2:
                raise ValueError(
                    f"Encoded caption at {sample.encoded_path} has unexpected shape {prompt_embeds.shape}"
                )

            sample_dict["prompt_embeds"] = prompt_embeds
        elif self.text_pre_encoded and not prompt_text:
            raise RuntimeError("Encoded caption path missing for sample with pre-encoded text.")

        return sample_dict


def cycle(dl):
    while True:
        for data in dl:
            yield data
