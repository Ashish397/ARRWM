"""Paired (clean + counterfactual) trajectory dataset for action-forcing ODE distillation.

Each sample pairs a ``clean_root/<name>.pt`` file with its counterfactual twin
``cf_root/<name>.pt``. Both were produced by ``gen_lmdb.py`` (see
``claude/gen_lmdb_v14.md``) with *identical* ``noise_seed`` so the only
source of divergence between them is the action edit.

The dataset emits the 21-frame truly-clean context latent
``clean_x_gt = zarr_latents[o : o+21]`` (fed to the DiT as ``clean_x`` for
teacher-forced attention). The regression target is the teacher's final
x0 snapshot, drawn from the per-branch ``trajectory`` tensor — the
dataset itself no longer reads or returns ``target_gt``.

The ride-frame contract: the teacher ran with
``clean_x = zarr[o:o+21]`` and denoised a 21-frame noisy branch at
``zarr[o+3 : o+24]`` under the 3-frame temporal shift; we only need the
first half of that (the context) here.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import zarr as zarr_lib
from torch.utils.data import Dataset


log = logging.getLogger(__name__)

NUM_FRAMES = 21
CONTEXT_FRAMES = 3
# Only the 21 preceding context latents are now required (target comes
# from the stored trajectory, not zarr).
STREAM_LATENT_SPAN = NUM_FRAMES                     # 21


def cycle(dl):
    """Infinite iterator over a DataLoader."""
    while True:
        for batch in dl:
            yield batch


def _fix_u6ej_path(zp: str) -> str:
    """Mirror gen_lmdb._fix_u6ej_path: old manifests used /projects/u6ej/... paths."""
    if "u6ej" in zp:
        return zp.replace(
            "/projects/u6ej/fbots/frodobots_encoded",
            "/projects/u6ex/fbots/frodobots_encoded",
        )
    return zp


def _build_ts_to_caption_json(caption_root: Path) -> Dict[str, Path]:
    """Map ride_ts -> encoded.json path under ``caption_root``.

    Caption trees are organised as::

        <caption_root>/output_rides_N/ride_<id>_<ts>/<ride_dir_name>_..._encoded.json

    Mirrors the logic in ``utils.eval_chain._build_ts_to_ride_dir``.
    """
    mapping: Dict[str, Path] = {}
    if not caption_root.exists():
        log.warning("caption_root does not exist: %s", caption_root)
        return mapping
    for out_dir in caption_root.iterdir():
        if not out_dir.is_dir():
            continue
        for ride_dir in out_dir.iterdir():
            if not ride_dir.is_dir():
                continue
            ts = ride_dir.name.split("_")[-1]
            enc = ride_dir / f"{ride_dir.name}_video_captions_OpenGVLab_InternVL3_8B_encoded.json"
            if enc.exists():
                mapping.setdefault(ts, enc)
    return mapping


def _load_prompt_embeds(enc_path: Path) -> torch.Tensor:
    """Load T5 caption embeddings from an ``*_encoded.json`` file.

    Returns a float32 CPU tensor of shape ``[512, 4096]``.
    """
    with open(enc_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    arr = data.get("caption_encoded")
    if arr is None:
        raise RuntimeError(f"'caption_encoded' missing in {enc_path}")
    return torch.tensor(arr, dtype=torch.float32)


class _ZarrGroupCache:
    """Process-local cache of opened zarr groups (one entry per ride)."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def get(self, zarr_path: str):
        g = self._cache.get(zarr_path)
        if g is None:
            g = zarr_lib.open_group(zarr_path, mode="r")
            self._cache[zarr_path] = g
        return g


class _SkipSample(Exception):
    """Raised inside ``_load_sample`` when a per-sample check fails.

    ``tag`` identifies *which* check tripped so the outer retry loop can
    keep a per-tag consecutive-failure counter. The retry loop converts
    this into a soft skip-to-next-sample; only when the same tag trips
    ``max_consecutive_same_fail`` times in a row (or the total number
    of consecutive skips exceeds a hard safety cap) is it escalated to
    a fatal ``RuntimeError``.
    """

    def __init__(self, tag: str, msg: str) -> None:
        super().__init__(msg)
        self.tag = tag


class PairedTrajectoryDataset(Dataset):
    """Pairs each clean ``.pt`` with its counterfactual twin.

    Args:
        clean_root: directory containing the clean ``*.pt`` files.
        cf_root: directory containing counterfactual ``*.pt`` files (same
            filenames as ``clean_root``).
        caption_root: top-level captions directory (e.g.
            ``/projects/u6ex/fbots/frodobots_captions/train``). Used to look
            up pre-encoded T5 prompt embeds by ``ride_ts``.
        max_pair: optional cap on number of paired samples (for fast smoke
            tests).
        require_cf: if True (default), skip samples without a CF twin.

    Returns per item (all CPU, float32 unless noted):
      - ``trajectory_clean``:   ``[7, 21, 16, 60, 104]`` fp32 (converted from
        fp16 storage); snapshot index ``0`` is pure noise drawn from the
        paired seed, index ``6`` == teacher_x0.
      - ``trajectory_cf``:      same shape, CF branch.
      - ``z_noisy``:            ``[21, 2]`` fp32 — z2/z7 for the noisy branch
        of the clean pass.
      - ``z_noisy_cf``:         ``[21, 2]`` fp32 — z2/z7 for the noisy branch
        of the CF pass.
      - ``z_clean``:            ``[21, 2]`` fp32 — z2/z7 for the clean context
        (identical between clean and CF; we assert).
      - ``clean_x_gt``:         ``[21, 16, 60, 104]`` fp32 — zarr_latents[o:o+21].
      - ``prompt_embeds``:      ``[512, 4096]`` fp32 — pre-encoded T5.
      - ``meta``:               dict with ``ride_ts``, ``window_offset``,
        ``noise_seed``, ``zarr_path``, ``city``.
    """

    def __init__(
        self,
        clean_root: str,
        cf_root: str,
        caption_root: str,
        *,
        max_pair: Optional[int] = None,
        require_cf: bool = True,
        allow_cf_fallback: bool = False,
        max_consecutive_same_fail: int = 5,
        max_consecutive_skips_total: int = 64,
    ) -> None:
        super().__init__()
        self.clean_root = Path(clean_root)
        self.cf_root = Path(cf_root)
        self.caption_root = Path(caption_root)
        self.allow_cf_fallback = bool(allow_cf_fallback)
        # Soft-fail bookkeeping: per-worker counters for consecutive
        # skip-to-next-sample events. On any successful load the counters
        # reset. A systematic data issue (same check tripping N times in a
        # row, or total consecutive skips exceeding a hard cap) escalates
        # to a fatal error so we don't silently burn GPU-hours.
        self._max_consec_same_fail = int(max_consecutive_same_fail)
        self._max_consec_total = int(max_consecutive_skips_total)
        if self._max_consec_same_fail < 1:
            raise ValueError("max_consecutive_same_fail must be >= 1")
        if self._max_consec_total < self._max_consec_same_fail:
            raise ValueError(
                "max_consecutive_skips_total must be >= max_consecutive_same_fail"
            )
        self._consec_fails_by_tag: Dict[str, int] = {}
        self._consec_fails_total: int = 0
        if not self.clean_root.is_dir():
            raise FileNotFoundError(f"clean_root not found: {self.clean_root}")
        if not self.cf_root.is_dir():
            raise FileNotFoundError(f"cf_root not found: {self.cf_root}")

        clean_files = sorted(p.name for p in self.clean_root.glob("*.pt"))
        paired: List[str] = []
        for name in clean_files:
            cf_path = self.cf_root / name
            if cf_path.exists():
                paired.append(name)
            elif not require_cf:
                paired.append(name)
        if max_pair is not None:
            paired = paired[: int(max_pair)]
        self._filenames: List[str] = paired
        if not self._filenames:
            raise RuntimeError(
                f"No paired .pt files found (clean_root={self.clean_root} "
                f"cf_root={self.cf_root})."
            )
        log.info(
            "PairedTrajectoryDataset: %d paired samples (clean_root=%s).",
            len(self._filenames), self.clean_root,
        )

        # Caption lookup is process-local and lazily built on first access.
        self._ts_to_caption: Optional[Dict[str, Path]] = None
        self._prompt_cache: Dict[str, torch.Tensor] = {}
        self._zarr_cache = _ZarrGroupCache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _caption_map(self) -> Dict[str, Path]:
        if self._ts_to_caption is None:
            self._ts_to_caption = _build_ts_to_caption_json(self.caption_root)
            log.info(
                "PairedTrajectoryDataset: indexed %d ride captions under %s",
                len(self._ts_to_caption), self.caption_root,
            )
        return self._ts_to_caption

    def _get_prompt(self, ride_ts: str) -> torch.Tensor:
        pe = self._prompt_cache.get(ride_ts)
        if pe is not None:
            return pe
        enc = self._caption_map().get(ride_ts)
        if enc is None:
            raise FileNotFoundError(
                f"No caption json found for ride_ts={ride_ts!r} under {self.caption_root}"
            )
        pe = _load_prompt_embeds(enc)
        self._prompt_cache[ride_ts] = pe
        return pe

    def _load_zarr_window(self, zarr_path: str, offset: int) -> torch.Tensor:
        """Return ``[offset : offset + STREAM_LATENT_SPAN]`` as fp32 CPU tensor.

        Raises ``_SkipSample("zarr_load")`` on any zarr I/O failure or
        short-read so the outer retry loop can advance to the next sample.
        """
        zarr_path_fixed = _fix_u6ej_path(zarr_path)
        try:
            g = self._zarr_cache.get(zarr_path_fixed)
            end = offset + STREAM_LATENT_SPAN
            lat_np = g["latents"][offset:end]
        except Exception as e:
            raise _SkipSample(
                "zarr_load",
                f"zarr read failed for {zarr_path_fixed}@{offset}: {e!r}",
            ) from e
        if lat_np.shape[0] != STREAM_LATENT_SPAN:
            raise _SkipSample(
                "zarr_short_read",
                f"zarr {zarr_path_fixed} returned {lat_np.shape[0]} latents at "
                f"offset {offset}; expected {STREAM_LATENT_SPAN}.",
            )
        return torch.from_numpy(lat_np.astype(np.float32))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._filenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """DataLoader-facing entry point with soft-fail + retry.

        Per-sample checks inside ``_load_sample`` raise ``_SkipSample``
        with a stable ``tag`` identifying the failure class. We catch
        those, log, bump a per-tag + global consecutive counter, and
        advance to the next index. On any successful load the counters
        reset. Systematic data issues escalate:

        * same ``tag`` N times in a row → ``RuntimeError`` (N =
          ``max_consecutive_same_fail``, default 5).
        * total consecutive skips ≥ hard cap → ``RuntimeError``.
        * every file in the dataset failed → ``RuntimeError``.
        """
        n = len(self._filenames)
        if n == 0:
            raise RuntimeError("PairedTrajectoryDataset is empty.")
        attempts = 0
        cur = int(idx) % n
        last_exc: Optional[BaseException] = None
        while attempts < n:
            try:
                sample = self._load_sample(cur)
            except _SkipSample as e:
                last_exc = e
                tag = e.tag
                cnt = self._consec_fails_by_tag.get(tag, 0) + 1
                self._consec_fails_by_tag[tag] = cnt
                self._consec_fails_total += 1
                fname = self._filenames[cur] if cur < n else "<oob>"
                log.warning(
                    "PairedTrajectoryDataset: skipping %s (idx=%d) tag=%s: %s "
                    "[consec_same=%d consec_total=%d]",
                    fname, cur, tag, e,
                    cnt, self._consec_fails_total,
                )
                if cnt >= self._max_consec_same_fail:
                    raise RuntimeError(
                        f"Dataset check '{tag}' tripped {cnt} samples in a row "
                        f"(cap={self._max_consec_same_fail}). Fix the data "
                        "before resuming."
                    ) from e
                if self._consec_fails_total >= self._max_consec_total:
                    raise RuntimeError(
                        f"Dataset accumulated {self._consec_fails_total} "
                        f"consecutive skips (cap={self._max_consec_total}); "
                        "aborting."
                    ) from e
                cur = (cur + 1) % n
                attempts += 1
                continue
            # Success — clear consecutive-failure state.
            if self._consec_fails_total > 0:
                log.info(
                    "PairedTrajectoryDataset: recovered after %d skips; "
                    "resetting failure counters.",
                    self._consec_fails_total,
                )
            self._consec_fails_by_tag.clear()
            self._consec_fails_total = 0
            return sample
        raise RuntimeError(
            f"PairedTrajectoryDataset: exhausted all {n} samples without a "
            "successful load."
        ) from last_exc

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Per-index sample loader. Raises ``_SkipSample`` on any
        recoverable issue so the outer retry loop can skip to the next
        index. Unrecoverable / out-of-contract errors still propagate.
        """
        fname = self._filenames[idx]
        try:
            clean_pt = torch.load(
                self.clean_root / fname, map_location="cpu", weights_only=False
            )
        except Exception as e:
            raise _SkipSample(
                "clean_pt_load", f"torch.load failed for clean/{fname}: {e!r}"
            ) from e
        try:
            cf_pt = torch.load(
                self.cf_root / fname, map_location="cpu", weights_only=False
            )
        except Exception as e:
            raise _SkipSample(
                "cf_pt_load", f"torch.load failed for cf/{fname}: {e!r}"
            ) from e

        # Paired-noise-seed sanity check: any mismatch means the producer
        # ran a different seed for the two branches, which would corrupt
        # the CF supervision signal.
        if int(clean_pt["noise_seed"]) != int(cf_pt["noise_seed"]):
            raise _SkipSample(
                "noise_seed_mismatch",
                f"noise_seed mismatch for {fname}: clean={clean_pt['noise_seed']}"
                f" cf={cf_pt['noise_seed']}",
            )
        if int(clean_pt["window_offset"]) != int(cf_pt["window_offset"]):
            raise _SkipSample(
                "window_offset_mismatch",
                f"window_offset mismatch for {fname}",
            )
        # zarr_path MUST match: both shards reference the same source
        # ride's encoded latents. A mismatch here would mean ``clean_x_gt``
        # and the CF trajectory come from different physical videos,
        # which would silently poison the CF supervision signal.
        if str(clean_pt["zarr_path"]) != str(cf_pt["zarr_path"]):
            raise _SkipSample(
                "zarr_path_mismatch",
                f"zarr_path mismatch for {fname}: "
                f"clean={clean_pt['zarr_path']!r} cf={cf_pt['zarr_path']!r}",
            )

        ride_ts = str(clean_pt["ride_ts"])
        offset = int(clean_pt["window_offset"])
        zarr_path = str(clean_pt["zarr_path"])

        traj_clean = clean_pt["trajectory"].to(torch.float32).contiguous()
        traj_cf = cf_pt["trajectory"].to(torch.float32).contiguous()
        if traj_clean.shape != traj_cf.shape:
            raise _SkipSample(
                "trajectory_shape_mismatch",
                f"trajectory shape mismatch for {fname}: "
                f"clean={tuple(traj_clean.shape)} cf={tuple(traj_cf.shape)}",
            )

        z_clean = clean_pt["z_clean"].to(torch.float32)
        z_noisy = clean_pt["z_noisy"].to(torch.float32)
        # CF file must carry the edited-action field. A silent fall-back
        # to clean.z_noisy would make the CF branch identical to the
        # clean branch, killing CF supervision without any error.
        if "z_noisy_cf" in cf_pt:
            z_noisy_cf = cf_pt["z_noisy_cf"].to(torch.float32)
        elif self.allow_cf_fallback:
            log.warning(
                "CF pt %s missing 'z_noisy_cf'; using clean z_noisy "
                "(allow_cf_fallback=True).", fname,
            )
            z_noisy_cf = cf_pt["z_noisy"].to(torch.float32)
        else:
            raise _SkipSample(
                "missing_z_noisy_cf",
                f"CF pt {fname} is missing 'z_noisy_cf'. Pass "
                "allow_cf_fallback=True (or fix the pipeline) to proceed.",
            )

        # z_clean in the CF file must match the clean-branch z_clean.
        # Both shards reference the same physical ride's true actions;
        # a mismatch here would mean the CF pt was regenerated against
        # a different source video than the clean pt (e.g. a stale
        # clean_root + fresh cf_root combo) and would silently poison
        # the shared context-action routing.
        if "z_clean" not in cf_pt:
            raise _SkipSample(
                "missing_z_clean_cf",
                f"CF pt {fname} is missing 'z_clean'; cannot validate "
                "against clean-branch context action.",
            )
        z_clean_cf_side = cf_pt["z_clean"].to(torch.float32)
        if z_clean.shape != z_clean_cf_side.shape:
            raise _SkipSample(
                "z_clean_shape_mismatch",
                f"z_clean shape mismatch for {fname}: "
                f"clean={tuple(z_clean.shape)} cf={tuple(z_clean_cf_side.shape)}",
            )
        if not torch.allclose(z_clean, z_clean_cf_side, atol=1e-6, rtol=0.0):
            max_abs = float((z_clean - z_clean_cf_side).abs().max())
            raise _SkipSample(
                "z_clean_value_mismatch",
                f"z_clean mismatch between clean and CF pt files for "
                f"{fname}: max|diff|={max_abs:.3e}. clean_root and "
                "cf_root appear to reference different source data.",
            )

        # The whole point of the CF branch is to edit the action signal.
        # If ``z_noisy_cf`` ever equals ``z_noisy`` exactly the CF pass
        # degenerates into a duplicate clean pass.
        if torch.equal(z_noisy_cf, z_noisy):
            raise _SkipSample(
                "z_noisy_cf_equals_z_noisy",
                f"z_noisy_cf == z_noisy for {fname}: CF pt has an "
                "*unedited* action signal, making the CF branch a no-op.",
            )

        # On-the-fly context fetch from zarr (21 preceding clean frames).
        # ``_load_zarr_window`` already raises ``_SkipSample`` on I/O or
        # short-read failures.
        zarr_win = self._load_zarr_window(zarr_path, offset)       # [21, C, H, W]
        clean_x_gt = zarr_win[:NUM_FRAMES].contiguous()            # [21, C, H, W]

        # Pre-encoded T5 prompt embeds. Missing captions or malformed
        # json should skip rather than abort the entire run.
        try:
            prompt_embeds = self._get_prompt(ride_ts)
        except FileNotFoundError as e:
            raise _SkipSample(
                "missing_prompt_caption",
                f"no caption json for ride_ts={ride_ts!r} ({fname}): {e}",
            ) from e
        except Exception as e:
            raise _SkipSample(
                "prompt_load_error",
                f"prompt load failed for ride_ts={ride_ts!r} ({fname}): {e!r}",
            ) from e

        meta = {
            "filename": fname,
            "ride_ts": ride_ts,
            "window_offset": offset,
            "noise_seed": int(clean_pt["noise_seed"]),
            "zarr_path": zarr_path,
            "city": str(clean_pt.get("city", "")),
        }

        return {
            "trajectory_clean": traj_clean,
            "trajectory_cf": traj_cf,
            "z_noisy": z_noisy,
            "z_noisy_cf": z_noisy_cf,
            "z_clean": z_clean,
            "clean_x_gt": clean_x_gt,
            "prompt_embeds": prompt_embeds,
            "meta": meta,
        }
