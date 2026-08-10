"""Chunk-level dataset over the CHAINED 14e pilot LMDBs (gen_lmdb_14e.py).

Maps the new chained format onto the EXISTING ODERegression interface
(trajectory [T_snap, 21, C, H, W] + z [21,2] + clean_x_gt [21, C, H, W])
so af_trainer/ode.py trains unchanged. See flow_viz/ODE_CAMPAIGN_STATE.md.

Stored file (one per (context, action-variant)):
  trajectory [GEN_CHUNKS=6, 5, 3, C, H, W] fp16  (5 snapshots per gen chunk,
      rungs t=[1000,625,357,208]+final on the 20-step shift-5 grid;
      slot -1 = committed), z [27, 2] in (throttle, steer) order (14e
      convention; legacy z2/z7 names retired), zarr_path, window_offset,
      noise_seed.
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
# Target chunks: clean_x must be the window shifted back ONE chunk (v14
# contract: mask context_shift=1 + rope_offset assume clean = noisy - 1
# chunk). c=3 would need zarr[off-3:off]; restrict to {4,5} instead.
TARGETS = (4, 5)


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
        # ODE_RANDOM_CF=1: sample the counterfactual branch uniformly at
        # random per __getitem__ instead of a fixed round-robin pairing.
        # With a multi-action LMDB (e.g. dir4) this matches ALL pairwise
        # action distances of the fan in expectation over steps — the
        # action-distribution form of the separation loss.
        self.random_cf = bool(os.environ.get("ODE_RANDOM_CF"))
        # ODE_ALLDIR=1: emit ONE sample PER cf-variant, consecutively, and
        # record group boundaries — with the trainer's grouped batch
        # sampler every optimizer step then contains ALL directions of one
        # (context, target-chunk) simultaneously.
        self.alldir = bool(os.environ.get("ODE_ALLDIR"))
        self.samples = []                  # (clean_file, cf_file, chunk, [pool])
        self.group_sizes = []
        # Curriculum bookkeeping: which DIRECTION each sample trains and
        # which group (context, target-chunk) it belongs to. Used by the
        # hard-direction sampler to re-select per group after epoch 1.
        self.sample_dir = []               # e.g. "cF" / "cBL" / "gt"
        self.sample_group = []             # group index
        for key, files in sorted(groups.items()):
            gt = [f for f in files if f.endswith("_gt.pt")]
            # No GT chain (pure-compass sets): prefer the NO-OP chain as the
            # clean side so every pair is (no-op, action) and the full 8-dir
            # fan stays trained (dir8n); else fall back to the first file.
            noop = [f for f in files if f.endswith("_cN.pt")]
            clean = gt[0] if gt else (noop[0] if noop else files[0])
            others = [f for f in files if f != clean]
            # ROLLOUT: a sample is a whole CHAIN, not a (chunk, variant)
            # pair -- _rollout_item ignores c, so emitting both TARGETS
            # would duplicate every chain byte-for-byte.
            _targets = (TARGETS[0],) if os.environ.get("ODE_ROLLOUT") else TARGETS
            for i, c in enumerate(_targets):
                if self.alldir and others:
                    grp = list(others)
                    if os.environ.get("ODE_ALLDIR_PAD8") and len(grp) == 7:
                        grp = grp + [clean]     # pad to 8 with the clean pair
                    gidx = len(self.group_sizes)
                    for cf in grp:
                        self.samples.append((clean, cf, c, others))
                        self.sample_dir.append(
                            re.sub(r".*_([a-zA-Z0-9]+)\.pt$", r"\1",
                                   os.path.basename(cf)))
                        self.sample_group.append(gidx)
                    self.group_sizes.append(len(grp))
                else:
                    cf = others[i % len(others)] if others else clean
                    self.samples.append((clean, cf, c, others if others else [clean]))
                    self.sample_dir.append(
                        re.sub(r".*_([a-zA-Z0-9]+)\.pt$", r"\1",
                               os.path.basename(cf)))
                    self.sample_group.append(len(self.group_sizes))
                    self.group_sizes.append(1)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _window(pt: Dict[str, Any], c: int, seed_lat: torch.Tensor):
        """-> (traj [5, 21, C, H, W], clean_x [21, C, H, W], frame_lo).

        clean_x follows the v14 teacher-forcing contract: the CLEAN window
        is the noisy window shifted back ONE chunk (global chunks c-4..c+2),
        so the noisy target block never sees its own answer through the
        clean-attention path (reviewed label-leak fix). Requires c >= 4.
        """
        assert c >= 4, f"target chunk {c} needs a one-chunk clean lead-in"
        traj = pt["trajectory"].to(torch.float32)          # [6, 5, 3, C, H, W]
        committed = traj[:, -1]                            # [6, 3, C, H, W]
        # global chunk list: 3 seed + generated 0..c-1 (committed)
        chunks = [seed_lat[i * NFB:(i + 1) * NFB] for i in range(3)] + \
                 [committed[g] for g in range(c)]
        ctx = torch.cat(chunks[-CTX_CHUNKS:], dim=0)       # [18, C, H, W]
        snap = traj[c]                                     # [5, 3, C, H, W]
        t_snap = snap.shape[0]
        traj_win = torch.cat(
            [ctx.unsqueeze(0).expand(t_snap, -1, -1, -1, -1), snap], dim=1)
        clean_x = torch.cat(chunks[-(CTX_CHUNKS + 1):], dim=0)  # [21, C, H, W]
        # chunks list covers global chunks 0..c+2 (3 seed + c gen); the last
        # CTX_CHUNKS+1 = 7 entries are global chunks c-4..c+2 = the clean
        # window (one chunk BEHIND the noisy window c-3..c+3).
        frame_lo = NFB * (c - 3)                           # global frame of noisy window start
        return traj_win, clean_x, frame_lo

    def _rollout_item(self, clean_f, cf_f, c, idx):
        """Items for the KV-cache ROLLOUT stage (af_model/ode_rollout.py).

        The teacher generated each chain as: SEED_CHUNKS real chunks written
        into the cache, then gen_chunks chunks produced autoregressively. To
        train the student the same way we need exactly that: the REAL seed, the
        full per-frame action stream over seed+generated, and every committed
        chunk as the target. No 21-frame window, no clean_x, no single target
        chunk -- those belong to the teacher-forced stage this replaces.
        """
        cpt = torch.load(clean_f, map_location="cpu", weights_only=False)
        fpt = cpt if (cf_f == clean_f) else \
            torch.load(cf_f, map_location="cpu", weights_only=False)
        zp, off = str(cpt["zarr_path"]), int(cpt["window_offset"])
        seed_chunks = int(cpt.get("seed_chunks", 3))
        seed_f = NFB * seed_chunks
        seed_lat = self.zarr_loader(zp, off, off + seed_f).to(torch.float32)
        ride_ts = os.path.basename(zp).replace(".zarr", "")
        out = {"seed_lat": seed_lat,
               "prompt_embeds": self.prompt_loader(ride_ts),
               "sample_idx": torch.tensor(int(idx), dtype=torch.long),
               "meta": {"filename": os.path.basename(clean_f),
                        "ride_ts": ride_ts, "window_offset": off,
                        "target_chunk": int(c), "zarr_path": zp,
                        "seed_chunks": seed_chunks,
                        "noise_seed": int(cpt["noise_seed"]), "city": ""}}
        for tag, pt in (("clean", cpt), ("cf", fpt)):
            traj = pt["trajectory"].to(torch.float32)      # [n_chunks,5,3,C,H,W]
            out[f"committed_{tag}"] = traj[:, -1]          # [n_chunks,3,C,H,W]
            out[f"z_{tag}"] = pt["z"].to(torch.float32)    # [seed_f+gen_f, 2]
            out[f"noise_seed_{tag}"] = torch.tensor(
                int(pt["noise_seed"]), dtype=torch.long)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clean_f, cf_f, c, cf_pool = self.samples[idx]
        if os.environ.get("ODE_ROLLOUT"):
            return self._rollout_item(clean_f, cf_f, c, idx)
        if self.random_cf and len(cf_pool) > 1:
            import random as _rnd
            cf_f = _rnd.choice(cf_pool)
        cpt = torch.load(clean_f, map_location="cpu", weights_only=False)
        fpt = cpt if (cf_f == clean_f) else \
            torch.load(cf_f, map_location="cpu", weights_only=False)
        zp, off = str(cpt["zarr_path"]), int(cpt["window_offset"])
        seed_lat = self.zarr_loader(zp, off, off + 3 * NFB).to(torch.float32)

        traj_c, clean_x, lo = self._window(cpt, c, seed_lat)
        traj_f, clean_x_cf, _ = self._window(fpt, c, seed_lat)   # cf gets ITS OWN chain's clean window
        z_c = cpt["z"].to(torch.float32)[lo:lo + NUM_FRAMES]
        z_f = fpt["z"].to(torch.float32)[lo:lo + NUM_FRAMES]
        # clean-branch actions follow the clean window (shifted back 1 chunk)
        z_clean_w = cpt["z"].to(torch.float32)[lo - NFB:lo - NFB + NUM_FRAMES]

        ride_ts = os.path.basename(zp).replace(".zarr", "")
        return {
            "trajectory_clean": traj_c,
            "trajectory_cf": traj_f,
            "z_noisy": z_c,
            "z_noisy_cf": z_f,
            "z_clean": z_clean_w,
            "clean_x_gt": clean_x,
            "clean_x_gt_cf": clean_x_cf,
            "prompt_embeds": self.prompt_loader(ride_ts),
            "sample_idx": torch.tensor(int(idx), dtype=torch.long),
            "meta": {"filename": os.path.basename(clean_f), "ride_ts": ride_ts,
                     "window_offset": off, "target_chunk": c,
                     "noise_seed": int(cpt["noise_seed"]), "zarr_path": zp,
                     "city": ""},
        }
