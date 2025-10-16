# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from torch.utils.data import Dataset
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
try:
    import datasets
except ImportError:
    datasets = None



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

    def __init__(self, latent_root: str, caption_root: str, num_frames: int = 21):
        self.latent_root = Path(latent_root)
        self.caption_root = Path(caption_root)
        self.num_frames = num_frames
        self.samples: list[tuple[Path, str]] = []

        if not self.latent_root.exists():
            raise FileNotFoundError(f"Latent root does not exist: {latent_root}")
        if not self.caption_root.exists():
            raise FileNotFoundError(f"Caption root does not exist: {caption_root}")

        latent_map: dict[Path, list[Path]] = {}
        for latent_path in sorted(self.latent_root.rglob("encoded_video_*.pt")):
            if not latent_path.is_file():
                continue
            try:
                rel_dir = latent_path.parent.relative_to(self.latent_root)
            except ValueError:
                continue
            latent_map.setdefault(rel_dir, []).append(latent_path)

        caption_map: dict[Path, list[Path]] = {}
        for caption_path in sorted(self.caption_root.rglob("*.json")):
            if not caption_path.is_file():
                continue
            parent = caption_path.parent
            try:
                rel_dir = parent.relative_to(self.caption_root)
            except ValueError:
                continue
            caption_map.setdefault(rel_dir, []).append(caption_path)

        all_rel_dirs = sorted(set(latent_map.keys()) | set(caption_map.keys()))

        for rel_dir in all_rel_dirs:
            latent_paths = sorted(latent_map.get(rel_dir, []))
            if not latent_paths:
                print(f"{rel_dir}: video doesn't exist, skipping")
                continue

            caption_candidates = caption_map.get(rel_dir, [])
            if not caption_candidates:
                print(f"{rel_dir}: caption doesn't exist, skipping")
                continue

            caption_path = self._select_caption_path(rel_dir, caption_candidates)
            if caption_path is None:
                print(f"{rel_dir}: caption doesn't exist, skipping")
                continue

            try:
                with caption_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"{caption_path}: failed to load caption ({exc}), skipping")
                continue

            caption = (data.get('combined_analysis') or '').strip()
            if not caption:
                print(f"{caption_path}: combined_analysis missing, skipping")
                continue

            for latent_path in latent_paths:
                self.samples.append((latent_path, caption))

        if not self.samples:
            raise RuntimeError('No paired latent/caption samples were found.')

    @staticmethod
    def _select_caption_path(rel_dir: Path, caption_paths: list[Path]):
        ride_name = rel_dir.name
        direct_name = f"{ride_name}.json"
        for candidate in caption_paths:
            if candidate.name == direct_name:
                return candidate

        prefix = f"{ride_name}_video_captions_"
        matches = [p for p in caption_paths if p.name.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import random

        latent_path, caption = self.samples[idx]
        latents = torch.load(latent_path, map_location='cpu')[0]
        if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
            raise ValueError(f'Unexpected latent format at {latent_path}')
        if latents.shape[0] == 0:
            raise ValueError(f'Latent sequence empty: {latent_path}')

        total = latents.shape[0]
        if total >= self.num_frames:
            start_max = total - self.num_frames
            start = random.randint(0, start_max) if start_max > 0 else 0
            latents = latents[start:start + self.num_frames]
        else:
            repeat = self.num_frames - total
            latents = torch.cat([latents, latents[-1:].repeat(repeat, 1, 1, 1)], dim=0)

        latents = latents.contiguous().float()
        return {
            'idx': idx,
            'prompts': caption,
            'real_latents': latents,
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
