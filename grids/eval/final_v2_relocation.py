"""Family-balanced ICLR ORB consensus on scenes with a complete local panel.

Uses the ORB/RANSAC calculation from the AAAI scene_consensus producer.
Real seed references follow each candidate's actual conditioning span.
At 15/30 s, output is scene-panel overlap only; the <50 relocation cutoff
has not been validated at those horizons.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fleet30s_common as fc  # noqa: E402
from final_v2_iclr import FAMILY, SEAT

orb = cv2.ORB_create(3000)


def real_indices(model):
    """Absolute real-seed references ending at shared boundary frame 32."""
    ctx = int(fc.ctx_of(model))
    if ctx <= 1:
        return (32,)
    local_first = max(0, ctx - 13)
    source_offset = 33 - ctx
    return tuple(np.linspace(source_offset + local_first, 32, 4).round().astype(int))


def frame_at(path, i):
    cap = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"undecodable frame {i}: {path}")
    return f


def descs_at(path, horizon_to_index):
    """Read every requested endpoint from one video handle."""
    cap = cv2.VideoCapture(
        str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    answer = {}
    try:
        for horizon, index in sorted(horizon_to_index.items(), key=lambda item: item[1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"undecodable frame {index}: {path}")
            answer[int(horizon)] = desc(frame)
    finally:
        cap.release()
    return answer


def load_model_descs(arguments):
    model, path, endpoint_map = arguments
    return model, descs_at(path, endpoint_map)


def desc(img):
    g = cv2.cvtColor(cv2.resize(img, (640, 352)), cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(g, None)
    points = np.float32([keypoint.pt for keypoint in keypoints]).reshape(-1, 2)
    return points, descriptors


def inl(kd0, kd1):
    (p0, d0), (p1, d1) = kd0, kd1
    if d0 is None or d1 is None or len(p0) < 8 or len(p1) < 8:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ms = matcher.match(d0, d1)
    if len(ms) < 8:
        return 0
    src = np.float32([p0[m.queryIdx] for m in ms])
    dst = np.float32([p1[m.trainIdx] for m in ms])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    # The calibrated direct-reference test is defined only at six seconds.
    # Longer horizons use the temporal break detector, so do not spend time
    # producing the explicitly unvalidated 15/30-second overlap diagnostic.
    e = pd.read_csv(out / "cpu_endpoints.csv")
    e = e[e.horizon_s.eq(6)].copy()
    complete_scenes = {scene for scene, g in e[e.horizon_s == 6].groupby("scene")
                       if set(SEAT.values()).issubset(set(g.model))}
    m = pd.read_csv(out / "video_manifest.csv")
    paths = m.set_index(["scene", "model"]).path.to_dict()
    dest = out / "relocation_panel_rows.csv"
    prior = pd.read_csv(dest) if dest.exists() else pd.DataFrame()
    if len(prior):
        prior = prior[
            prior.scene.isin(complete_scenes) & prior.horizon_s.eq(6)
        ].copy()
    prior = prior.rename(columns={"relocation_flag_6s": "relocation_flag_6s_exploratory"})
    if len(prior):
        prior.loc[prior.horizon_s == 6, "status"] = (
            f"{len(SEAT)}-family adaptation; candidate-aligned seed refs; "
            "cutoff calibration transfer unvalidated"
        )
    rows = prior.to_dict("records")
    pair_path = out / "relocation_pair_evidence.csv"
    if pair_path.exists():
        old_pairs = pd.read_csv(pair_path)
        pairs = old_pairs[old_pairs.scene.isin(complete_scenes)].to_dict("records")
    else:
        pairs = []

    # A finish pass may intentionally stop after rendering candidates for
    # human adjudication.  On resume, do not decode and hash every seed again
    # when the complete panel is already present.  This remains fail-closed:
    # the cached table must have the exact current endpoint key set, current
    # video hashes, model-specific real-reference indices, one real hash per
    # scene, and non-empty pair evidence for every endpoint key.
    current = e[e.scene.isin(complete_scenes)].copy()
    key_cols = ["scene", "model", "horizon_s"]
    expected_keys = set(map(tuple, current[key_cols].itertuples(index=False, name=None)))
    cached_keys = (set(map(tuple, prior[key_cols].itertuples(index=False, name=None)))
                   if len(prior) and set(key_cols).issubset(prior.columns) else set())
    cache_complete = bool(expected_keys) and cached_keys == expected_keys
    if cache_complete:
        cache_complete = not prior.duplicated(key_cols).any()
    if cache_complete:
        expected_hash = current.set_index(key_cols).video_sha256
        cached_hash = prior.set_index(key_cols).video_sha256
        cache_complete = expected_hash.sort_index().equals(cached_hash.sort_index())
    if cache_complete:
        cache_complete = all(
            str(row.reference_indices) == ",".join(map(str, real_indices(row.model)))
            for row in prior.itertuples()
        )
    if cache_complete:
        cache_complete = (
            prior.real_sha256.notna().all()
            and prior.groupby("scene").real_sha256.nunique().eq(1).all()
        )
    if cache_complete:
        old_pairs = pd.read_csv(pair_path)
        pair_keys = set(map(tuple, old_pairs[["scene", "model", "horizon_s"]]
                            .itertuples(index=False, name=None)))
        cache_complete = pair_keys == expected_keys
    if cache_complete:
        # Preserve the schema/status normalisation above even when resuming.
        prior.to_csv(dest, index=False)
        print(f"relocation cache validated rows {len(prior)} pairs {len(old_pairs)}")
        return

    current_scene = None
    real = None
    scene_kd = None
    seed_hash = None
    seed_hashes = {}
    relocation_workers = int(os.environ.get("PANEL32_RELOCATION_WORKERS", "1"))
    if relocation_workers < 1:
        raise ValueError("PANEL32_RELOCATION_WORKERS must be positive")
    completed_since_save = 0
    for (scene, horizon), group in e.groupby(["scene", "horizon_s"], sort=True):
        if scene not in complete_scenes:
            continue
        models = set(group.model)
        eligible_rows = [r for r in group.itertuples()
                         if (set(SEAT.values()) - {SEAT[FAMILY[r.model]]}).issubset(models)]
        if not eligible_rows:
            continue
        key = (scene, horizon)
        existing = prior[(prior.scene == scene) & (prior.horizon_s == horizon)] if len(prior) else pd.DataFrame()
        stamps = {r.model: r.video_sha256 for r in eligible_rows}
        expected_refs = {
            r.model: ",".join(map(str, real_indices(r.model))) for r in eligible_rows
        }
        seed_path = fc.SEED_CLIP(scene.rsplit("_", 1)[0])
        if seed_path not in seed_hashes:
            seed_hashes[seed_path] = sha(seed_path)
        expected_seed_hash = seed_hashes[seed_path]
        if len(existing) == len(eligible_rows) and set(existing.model) == set(stamps) and all(
            existing.set_index("model").loc[model, "video_sha256"] == digest
            for model, digest in stamps.items()) and (existing.real_sha256 == expected_seed_hash).all() and all(
            str(existing.set_index("model").loc[model, "reference_indices"]) == refs
            for model, refs in expected_refs.items()
        ):
            continue
        if scene != current_scene:
            seed_hash = expected_seed_hash
            real = {i: desc(frame_at(seed_path, i)) for i in (0, 4, 8, 12, 20, 24, 28, 32)}
            scene_rows = current[current.scene == scene]
            endpoint_maps = {
                model: {
                    int(item.horizon_s): int(item.endpoint_index)
                    for item in model_rows.itertuples()
                }
                for model, model_rows in scene_rows.groupby("model")
            }

            jobs = [
                (model, paths[(scene, model)], endpoint_maps[model])
                for model in sorted(endpoint_maps)
            ]
            # OpenCV/FFmpeg decoder state is not reliably thread-safe for
            # concurrent opens in one process.  Separate processes preserve
            # the exact descriptor calculation while allowing bounded I/O.
            if relocation_workers == 1:
                scene_kd = dict(map(load_model_descs, jobs))
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(relocation_workers, len(endpoint_maps))
                ) as executor:
                    scene_kd = dict(executor.map(load_model_descs, jobs))
            current_scene = scene
        rows = [x for x in rows if (x["scene"], x["horizon_s"]) != key]
        pairs = [x for x in pairs if (x["scene"], x["horizon_s"]) != key]
        needed = set()
        for r in eligible_rows:
            seats = SEAT.copy()
            seats[FAMILY[r.model]] = r.model
            needed.update(seats.values())
        kd = {model: scene_kd[model][int(horizon)] for model in needed}
        def score_candidate(r):
            seats = SEAT.copy()
            seats[FAMILY[r.model]] = r.model
            eligible = {name: kd[name] for name in seats.values() if name != r.model}
            refs = real_indices(r.model)
            eligible.update({f"real_{i}": real[i] for i in refs})
            values = {name: inl(kd[r.model], item) for name, item in eligible.items()}
            winner = max(values, key=values.get)
            row = dict(scene=scene, model=r.model, horizon_s=horizon,
                       panel_inliers=values[winner], best_peer=winner,
                       panel_members=json.dumps(sorted(eligible)),
                       panel_size=len(eligible),
                       relocation_flag_6s_exploratory=int(values[winner] < 50),
                       video_sha256=stamps[r.model], real_sha256=seed_hash,
                       reference_indices=",".join(map(str, refs)),
                       reference_schema="aligned_real_frame32_v3",
                       status=(f"{len(SEAT)}-family adaptation; candidate-aligned seed "
                               "refs; cutoff calibration transfer unvalidated"))
            pair_rows = [
                dict(scene=scene, model=r.model, horizon_s=horizon,
                     peer=name, inliers=value)
                for name, value in values.items()
            ]
            return row, pair_rows

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(relocation_workers, len(eligible_rows))
        ) as executor:
            scored = list(executor.map(score_candidate, eligible_rows))
        rows.extend(row for row, _ in scored)
        for _, pair_rows in scored:
            pairs.extend(pair_rows)
        completed_since_save += 1
        if completed_since_save >= 25:
            pd.DataFrame(rows).to_csv(dest, index=False)
            pd.DataFrame(pairs).to_csv(pair_path, index=False)
            completed_since_save = 0
        print(scene, horizon, len(eligible_rows), flush=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    pd.DataFrame(pairs).to_csv(pair_path, index=False)
    if rows:
        assert not pd.DataFrame(rows).duplicated(["scene", "model", "horizon_s"]).any()
    print("relocation rows", len(rows))


if __name__ == "__main__":
    main()
