"""Package one real seed and its prompt for a matched cross-project ODE probe.

Usage: python utils/export_ode_probe_seed.py <ARRWM root> <window index> <output.pt>
The package excludes teacher and student outputs; it only replaces a Zarr
read on another project where the source ride is unavailable.
"""
import json
import os
import sys

import torch

from utils.zarr_dataset import ZarrRideDataset


def main() -> None:
    arr, window_str, output = sys.argv[1:4]
    window = int(window_str)
    with open(os.path.join(arr, "analysis/eval_final/phaseA_windows.json")) as f:
        selected = json.load(f)[window]
    manifest_obj = torch.load(
        os.path.join(arr, "analysis/eval_final/manifest_unseen.pt"),
        map_location="cpu", weights_only=False,
    )
    rides = (manifest_obj.get("rides", manifest_obj)
             if isinstance(manifest_obj, dict) else manifest_obj)
    ride = next(r for r in rides if r["zarr_path"] == selected["zarr_path"])
    start = int(selected["offset"])
    seed = ZarrRideDataset.load_latent_chunk(
        selected["zarr_path"], start, start + 9).unsqueeze(0)
    prompt_embeds = ride["prompt_embeds"].unsqueeze(0)
    torch.save({
        "seed": seed,
        "prompt_embeds": prompt_embeds,
        "source_zarr": selected["zarr_path"],
        "source_offset": start,
    }, output)
    print(f"saved {output}: seed={tuple(seed.shape)}, prompt={tuple(prompt_embeds.shape)}")


if __name__ == "__main__":
    main()
