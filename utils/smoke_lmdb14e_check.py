"""Load-side check for the lmdb14e generation smoke.

For each smoke variant dir: torch.load every .pt, assert the stored shapes,
then run ChunkedODEDataset end-to-end (real zarr + caption loaders) and
print one sample's tensor shapes. Exits nonzero on any mismatch.
Env: SMOKE_ROOT (parent of the per-variant dirs).
"""
import os, sys, glob
import torch

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
sys.path[:0] = [ARR, f"{ARR}/action-forcing"]
ROOT = os.environ.get("SMOKE_ROOT", f"{ARR}/.lmdb14e_smoke")
CAPTIONS = "/projects/u6ex/fbots/frodobots_captions/train"

from af_utils.chunked_ode_dataset import ChunkedODEDataset
from af_utils.dataset import _load_prompt_embeds, _build_ts_to_caption_json
from utils.zarr_dataset import ZarrRideDataset

from pathlib import Path as _Path
cap = _build_ts_to_caption_json(_Path(CAPTIONS))
ok = True
for vdir in sorted(glob.glob(f"{ROOT}/*")):
    if not os.path.isdir(vdir):
        continue
    files = sorted(glob.glob(f"{vdir}/*.pt"))
    print(f"[check] {os.path.basename(vdir)}: {len(files)} files")
    for f in files:
        pt = torch.load(f, map_location="cpu", weights_only=False)
        tr = pt["trajectory"]
        if tuple(tr.shape) != (6, 5, 3, 16, 60, 104):
            print(f"  BAD trajectory shape {tuple(tr.shape)} in {f}"); ok = False
        if tuple(pt["z"].shape) != (27, 2):
            print(f"  BAD z shape {tuple(pt['z'].shape)} in {f}"); ok = False
        if not torch.isfinite(tr.float()).all():
            print(f"  NONFINITE trajectory in {f}"); ok = False
    ds = ChunkedODEDataset(
        root=vdir,
        zarr_loader=ZarrRideDataset.load_latent_chunk,
        prompt_loader=lambda ts: _load_prompt_embeds(cap[ts]))
    print(f"  dataset: {len(ds)} samples")
    s = ds[0]
    for k in ("trajectory_clean", "trajectory_cf", "z_noisy", "clean_x_gt", "prompt_embeds"):
        print(f"  {k}: {tuple(s[k].shape)}")
    assert tuple(s["trajectory_clean"].shape) == (5, 21, 16, 60, 104), "traj window shape"
    assert tuple(s["clean_x_gt"].shape) == (21, 16, 60, 104), "clean_x shape"
    assert tuple(s["z_noisy"].shape) == (21, 2), "z shape"
print("SMOKE-CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
