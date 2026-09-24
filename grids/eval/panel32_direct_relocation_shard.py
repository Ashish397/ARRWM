#!/usr/bin/env python3
"""Shard the final-v2 direct relocation calculation by scene/horizon group.

This is a scheduling wrapper around the exact descriptor, reference-panel, and
RANSAC primitives in :mod:`final_v2_relocation`.  Each scene/horizon group is
independent, so splitting those groups does not change any score.  Shards write
private CSVs and a separate merge mode validates exact key coverage before
publishing the filenames consumed by the final evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import final_v2_relocation as reloc


KEY = ["scene", "model", "horizon_s"]


def score_shard(out: Path, shard_index: int, shard_count: int) -> None:
    endpoints = pd.read_csv(out / "cpu_endpoints.csv")
    complete_scenes = {
        scene
        for scene, group in endpoints[endpoints.horizon_s == 6].groupby("scene")
        if set(reloc.SEAT.values()).issubset(set(group.model))
    }
    manifest = pd.read_csv(out / "video_manifest.csv")
    paths = manifest.set_index(["scene", "model"]).path.to_dict()
    groups = [
        (key, group)
        for key, group in endpoints.groupby(["scene", "horizon_s"], sort=True)
        if key[0] in complete_scenes
    ]
    selected = groups[shard_index::shard_count]
    rows: list[dict] = []
    pairs: list[dict] = []
    cached_scene = None
    real = None
    seed_hash = None
    for ordinal, ((scene, horizon), group) in enumerate(selected, 1):
        models = set(group.model)
        eligible_rows = [
            row
            for row in group.itertuples()
            if (set(reloc.SEAT.values()) - {reloc.SEAT[reloc.FAMILY[row.model]]})
            .issubset(models)
        ]
        if not eligible_rows:
            continue
        if scene != cached_scene:
            seed_path = reloc.fc.SEED_CLIP(scene.rsplit("_", 1)[0])
            seed_hash = reloc.sha(seed_path)
            real = {
                i: reloc.desc(reloc.frame_at(seed_path, i))
                for i in (0, 4, 8, 12, 20, 24, 28, 32)
            }
            cached_scene = scene
        stamps = {row.model: row.video_sha256 for row in eligible_rows}
        needed = set()
        for row in eligible_rows:
            seats = reloc.SEAT.copy()
            seats[reloc.FAMILY[row.model]] = row.model
            needed.update(seats.values())
        kd = {
            row.model: reloc.desc(reloc.frame_at(paths[(scene, row.model)], row.endpoint_index))
            for row in group.itertuples()
            if row.model in needed
        }
        for row in eligible_rows:
            seats = reloc.SEAT.copy()
            seats[reloc.FAMILY[row.model]] = row.model
            eligible = {name: kd[name] for name in seats.values() if name != row.model}
            refs = reloc.real_indices(row.model)
            eligible.update({f"real_{i}": real[i] for i in refs})
            values = {name: reloc.inl(kd[row.model], item) for name, item in eligible.items()}
            winner = max(values, key=values.get)
            rows.append({
                "scene": scene,
                "model": row.model,
                "horizon_s": horizon,
                "panel_inliers": values[winner],
                "best_peer": winner,
                "panel_members": json.dumps(sorted(eligible)),
                "panel_size": len(eligible),
                "relocation_flag_6s_exploratory": (
                    int(values[winner] < 50) if horizon == 6 else None
                ),
                "video_sha256": stamps[row.model],
                "real_sha256": seed_hash,
                "reference_indices": ",".join(map(str, refs)),
                "reference_schema": "aligned_real_frame32_v3",
                "status": (
                    f"{len(reloc.SEAT)}-family adaptation; candidate-aligned seed refs; "
                    "cutoff calibration transfer unvalidated"
                    if horizon == 6
                    else "long-horizon overlap diagnostic only"
                ),
            })
            pairs.extend(
                {
                    "scene": scene,
                    "model": row.model,
                    "horizon_s": horizon,
                    "peer": name,
                    "inliers": value,
                }
                for name, value in values.items()
            )
        print(
            f"[{ordinal}/{len(selected)}] shard={shard_index}/{shard_count} "
            f"{scene} {horizon} rows={len(eligible_rows)}",
            flush=True,
        )
    shard_dir = out / "direct_relocation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(shard_dir / f"rows_{shard_index:03d}.csv", index=False)
    pd.DataFrame(pairs).to_csv(shard_dir / f"pairs_{shard_index:03d}.csv", index=False)
    (shard_dir / f"shard_{shard_index:03d}.COMPLETE").write_text(
        f"rows={len(rows)} pairs={len(pairs)}\n"
    )


def merge_shards(out: Path, shard_count: int) -> None:
    shard_dir = out / "direct_relocation_shards"
    row_frames = []
    pair_frames = []
    for shard in range(shard_count):
        marker = shard_dir / f"shard_{shard:03d}.COMPLETE"
        if not marker.exists():
            raise FileNotFoundError(marker)
        row_frames.append(pd.read_csv(shard_dir / f"rows_{shard:03d}.csv"))
        pair_frames.append(pd.read_csv(shard_dir / f"pairs_{shard:03d}.csv"))
    rows = pd.concat(row_frames, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(
        KEY + ["peer"]
    ).reset_index(drop=True)
    endpoints = pd.read_csv(out / "cpu_endpoints.csv")
    expected = set(map(tuple, endpoints[KEY].itertuples(index=False, name=None)))
    actual = set(map(tuple, rows[KEY].itertuples(index=False, name=None)))
    if actual != expected:
        raise ValueError(
            f"direct relocation key mismatch missing={len(expected-actual)} "
            f"extra={len(actual-expected)}"
        )
    if rows.duplicated(KEY).any():
        raise ValueError("duplicate direct relocation rows")
    if set(map(tuple, pairs[KEY].itertuples(index=False, name=None))) != expected:
        raise ValueError("pair evidence does not cover every endpoint key")
    rows.to_csv(out / "relocation_panel_rows.csv", index=False)
    pairs.to_csv(out / "relocation_pair_evidence.csv", index=False)
    print(f"merged direct relocation rows={len(rows)} pairs={len(pairs)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge_shards(args.out.resolve(), args.shard_count)
    else:
        if args.shard_index is None or not 0 <= args.shard_index < args.shard_count:
            parser.error("--shard-index must be in [0, shard-count)")
        score_shard(args.out.resolve(), args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
