#!/usr/bin/env python3
"""Build per-ride forward<->rear latent alignment maps by WALL-CLOCK time.

WHY (honest): the front (uid_s_1000) and rear (uid_s_1001) cameras are NOT
frame-synchronised. Measured across 30 weu rides the rear stream starts -1.1..+6.6s
after the front and drifts a further -2.4..+4.0s over the ride; the rear is also VFR
(~10-20 fps). So forward latent i does NOT correspond to rear latent i. Naive
index alignment would feed the dual-view model temporally-mismatched pairs.

Both cameras DO carry per-frame wall-clock timestamps in the raw ride dir
(front_camera_timestamps_<rid>.csv / rear_camera_timestamps_<rid>.csv). This builds
a robust map: for each FORWARD latent we estimate its wall-clock (by interpolating
the raw per-frame wall-clock at the latent's fractional frame position, which also
absorbs the encode's block-overlap and the rear VFR), then pick the REAR latent
nearest in wall-clock. Forward latents outside the rear overlap window map to -1.

This also fixes the forward-trimming issue (some forward zarrs were action-window
trimmed): wall-clock matching uses true clock time, independent of trimming.

Output per ride: <rear_root>/<ride_ts>.align.npy  (int32, length = T_forward;
entry = rear latent index or -1) plus a JSON summary with alignment quality.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import zarr

STALE_PREFIX = "/home/ashish/frodobots/"
RAW_ROOT = "/lus/lfs1aip2/projects/u6ex/fbots/frodobots_raw/"


def remap(p: str) -> str:
    return p.replace(STALE_PREFIX, RAW_ROOT) if p.startswith(STALE_PREFIX) else p


def load_wallclock_csv(path: str) -> np.ndarray:
    ts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp"]))
    return np.asarray(ts, dtype=np.float64)


def per_latent_wallclock(wall: np.ndarray, n_latents: int) -> np.ndarray:
    """Estimate wall-clock for each of n_latents by interpolating the raw per-frame
    wall-clock at the latent's fractional frame position. Robust to VFR + the 4:1
    latent ratio + block overlap (self-calibrates to actual frame/latent count)."""
    if n_latents <= 1 or wall.size == 0:
        return np.full(n_latents, wall[0] if wall.size else 0.0, dtype=np.float64)
    frame_pos = np.linspace(0.0, wall.size - 1, n_latents)
    return np.interp(frame_pos, np.arange(wall.size), wall)


def build_one(ride_ts: str, forward_root: str, rear_root: str, ride_dir: str,
              tol: float) -> dict:
    fz = os.path.join(forward_root, f"{ride_ts}.zarr")
    rz = os.path.join(rear_root, f"{ride_ts}.zarr")
    rid = os.path.basename(ride_dir.rstrip("/")).split("_")[1]
    fcsv = glob.glob(os.path.join(ride_dir, "front_camera_timestamps_*.csv"))
    rcsv = glob.glob(os.path.join(ride_dir, "rear_camera_timestamps_*.csv"))
    out = {"ride_ts": ride_ts, "ok": False, "reason": ""}
    if not (os.path.isdir(fz) and os.path.isdir(rz)):
        out["reason"] = "missing zarr"; return out
    if not (fcsv and rcsv):
        out["reason"] = "missing csv"; return out

    Tf = int(zarr.open(fz, "r")["latents"].shape[0])
    Tr = int(zarr.open(rz, "r")["latents"].shape[0])
    WF = load_wallclock_csv(fcsv[0])
    WR = load_wallclock_csv(rcsv[0])

    wc_f = per_latent_wallclock(WF, Tf)
    wc_r = per_latent_wallclock(WR, Tr)

    # nearest rear latent per forward latent (wc_r is monotonic non-decreasing)
    idx = np.searchsorted(wc_r, wc_f)
    idx = np.clip(idx, 1, Tr - 1)
    left = idx - 1
    choose_left = np.abs(wc_f - wc_r[left]) <= np.abs(wc_f - wc_r[idx])
    j = np.where(choose_left, left, idx).astype(np.int32)
    resid = np.abs(wc_f - wc_r[j])

    lo, hi = wc_r[0] - tol, wc_r[-1] + tol
    outside = (wc_f < lo) | (wc_f > hi) | (resid > tol)
    j[outside] = -1

    n_aligned = int((j >= 0).sum())
    amap = j
    np.save(os.path.join(rear_root, f"{ride_ts}.align.npy"), amap)

    r_ok = resid[j >= 0]
    out.update(
        ok=True, Tf=Tf, Tr=Tr,
        n_aligned=n_aligned, frac_aligned=round(n_aligned / max(1, Tf), 3),
        resid_med=round(float(np.median(r_ok)), 3) if r_ok.size else None,
        resid_max=round(float(r_ok.max()), 3) if r_ok.size else None,
        start_off=round(float(wc_r[0] - wc_f[0]), 2),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward_root", default="/projects/u6ex/fbots/frodobots_encoded_weu")
    ap.add_argument("--rear_root", default="/projects/u6ex/fbots/frodobots_encoded_weu_rear")
    ap.add_argument("--csv", default="data/rear_encode/weu_rear_smoke.csv",
                    help="CSV (ride_ts, video_path, ride_dir) selecting which rides to align.")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="Max wall-clock residual (s) to accept a forward<->rear latent pair.")
    ap.add_argument("--summary_json", default="data/rear_encode/alignment_summary.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    results = []
    for r in rows:
        res = build_one(r["ride_ts"], args.forward_root, args.rear_root,
                        remap(r["ride_dir"]), args.tol)
        results.append(res)
        tag = "OK " if res["ok"] else "SKIP"
        extra = (f"Tf={res.get('Tf')} Tr={res.get('Tr')} aligned={res.get('frac_aligned')} "
                 f"resid_med={res.get('resid_med')} start_off={res.get('start_off')}"
                 if res["ok"] else res["reason"])
        print(f"[{tag}] {r['ride_ts']}  {extra}")

    ok = [r for r in results if r["ok"]]
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.summary_json, "w"), indent=2)
    if ok:
        fr = np.array([r["frac_aligned"] for r in ok])
        print(f"\n{len(ok)}/{len(results)} rides aligned. "
              f"frac_aligned: min={fr.min():.2f} median={np.median(fr):.2f} max={fr.max():.2f}")
    print(f"summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
