# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from torch.utils.data import Dataset
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


def cycle(dl):
    while True:
        for data in dl:
            yield data
