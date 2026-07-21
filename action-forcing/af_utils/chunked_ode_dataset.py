"""Chunk-level dataset over the CHAINED 14e pilot LMDBs (gen_lmdb_14e.py).

Maps the new chained format onto the EXISTING ODERegression interface
(trajectory [T_snap, 21, C, H, W] + z [21,2] + clean_x_gt [21, C, H, W])
so af_trainer/ode.py trains unchanged. See flow_viz/ODE_CAMPAIGN_STATE.md.

Stored file (one per (context, action-variant)):
  trajectory [GEN_CHUNKS=6, 5, 3, C, H, W] fp16  (5 snapshots per gen chunk,
      rungs t=[1000,625,357,208]+final on the 20-step shift-5 grid;
      slot -1 = committed), z [27, 2], zarr_path, window_offset, noise_seed.
  Seed context (3 chunks) is NOT stored — loaded from zarr.

Sample = target gen-chunk c in {3,4,5}; its 6 preceding chunks (real seed +
teacher-committed) form the 18 context frames, tiled across the 5 snapshot
slots; frames 18..20 are chunk c's snapshots. clean_x_gt uses the COMMITTED
context + committed target (self-context exposure — deliberate change from
the old GT-window semantics).

Variant pairing per (ride_ts, offset): clean = the 'gt' file when present
else the first variant; cf = a different variant file round-robin over the
remaining ones; single-variant sets duplicate clean as cf (train with
lambda_cf=0, precedent af_utils/dataset.py:179).
"""
from __future__ import annotations
import glob, os, re
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

NFB = 3
CTX_CHUNKS = 6
NUM_FRAMES = 21
TARGETS = (3, 4, 5)                       # gen-chunk indices with full context


class ChunkedODEDataset(Dataset):
    def __init__(self, root: str, zarr_loader, prompt_loader,
                 clean_only: bool = False):
        """zarr_loader(zarr_path, lo, hi) -> [hi-lo, C, H, W] float32;
        prompt_loader(ride_ts) -> [512, 4096]."""
        self.root = root
        self.zarr_loader = zarr_loader
        self.prompt_loader = prompt_loader
        self.clean_only = clean_only
        groups: Dict[str, List[str]] = {}
        for f in sorted(glob.glob(os.path.join(root, "*.pt"))):
            m = re.match(r"(.+_o\d{5})_([a-zA-Z0-9]+)\.pt$", os.path.basename(f))
            if m:
                groups.setdefault(m.group(1), []).append(f)
        self.samples = []                  # (clean_file, cf_file, target_chunk)
        for key, files in sorted(groups.items()):
            gt = [f for f in files if f.endswith("_gt.pt")]
            clean = gt[0] if gt else files[0]
            others = [f for f in files if f != clean]
            for i, c in enumerate(TARGETS):
                cf = others[i % len(others)] if others else clean
                self.samples.append((clean, cf, c))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _window(pt: Dict[str, Any], c: int, seed_lat: torch.Tensor):
        """-> (traj [5, 21, C, H, W], committed_ctx [21, C, H, W], frame_lo)."""
        traj = pt["trajectory"].to(torch.float32)          # [6, 5, 3, C, H, W]
        committed = traj[:, -1]                            # [6, 3, C, H, W]
        # global chunk list: 3 seed + generated 0..c-1, keep the last 6
        chunks = [seed_lat[i * NFB:(i + 1) * NFB] for i in range(3)] + \
                 [committed[g] for g in range(c)]
        ctx = torch.cat(chunks[-CTX_CHUNKS:], dim=0)       # [18, C, H, W]
        snap = traj[c]                                     # [5, 3, C, H, W]
        t_snap = snap.shape[0]
        traj_win = torch.cat(
            [ctx.unsqueeze(0).expand(t_snap, -1, -1, -1, -1), snap], dim=1)
        clean_x = torch.cat([ctx, committed[c]], dim=0)    # [21, C, H, W]
        frame_lo = NFB * (c - 3)                           # global frame of window start
        return traj_win, clean_x, frame_lo

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clean_f, cf_f, c = self.samples[idx]
        cpt = torch.load(clean_f, map_location="cpu", weights_only=False)
        fpt = cpt if (cf_f == clean_f) else \
            torch.load(cf_f, map_location="cpu", weights_only=False)
        zp, off = str(cpt["zarr_path"]), int(cpt["window_offset"])
        seed_lat = self.zarr_loader(zp, off, off + 3 * NFB).to(torch.float32)

        traj_c, clean_x, lo = self._window(cpt, c, seed_lat)
        traj_f, _, _ = self._window(fpt, c, seed_lat)
        z_c = cpt["z"].to(torch.float32)[lo:lo + NUM_FRAMES]
        z_f = fpt["z"].to(torch.float32)[lo:lo + NUM_FRAMES]

        ride_ts = os.path.basename(zp).replace(".zarr", "")
        return {
            "trajectory_clean": traj_c,
            "trajectory_cf": traj_f,
            "z_noisy": z_c,
            "z_noisy_cf": z_f,
            "z_clean": z_c,
            "clean_x_gt": clean_x,
            "prompt_embeds": self.prompt_loader(ride_ts),
            "meta": {"filename": os.path.basename(clean_f), "ride_ts": ride_ts,
                     "window_offset": off, "target_chunk": c,
                     "noise_seed": int(cpt["noise_seed"]), "zarr_path": zp,
                     "city": ""},
        }
