"""Detect persistent scene replacements across long autoregressive rollouts.

The six-second Reloc-Cal-v2 endpoint score cannot be applied directly at long
horizons: ordinary camera travel eventually removes all direct feature overlap
with the seed.  This producer instead searches for a *break* in a temporal
chain of geometrically verified correspondences.  A boundary is a relocation
candidate only when

1. no frame in the following one-second bank matches any frame in the
   preceding one-second bank;
2. both banks are internally coherent; and
3. the new bank contains enough features to rule out featureless collapse.

Consequently, gradual travel remains connected through intermediate views,
while a coherent replacement scene creates a persistent break.  The output is
cumulative by endpoint: once a replacement occurs, later endpoints remain
flagged.  This is a long-horizon companion to, rather than a reinterpretation
of, the frozen six-second Reloc-Cal-v2 result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


VERSION = "iclr-long-reloc-v2-chain4hz-bank1s-input-fingerprint"
ENDPOINTS = (6, 12, 18, 24, 30)
SIZE = (640, 352)


def resolve_path(path: str, local_prefix: str, remote_prefix: str) -> str:
    if os.path.exists(path):
        return path
    if local_prefix and path.startswith(local_prefix):
        candidate = remote_prefix + path[len(local_prefix):]
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(path)


def sample_plan(context: int, fps: float, frame_count: int,
                sample_hz: float) -> tuple[list[int], list[float]]:
    """Return the final real frame and a regular generated-time grid."""
    last_generated = frame_count - 1
    duration = (last_generated - context) / fps
    times = [-1.0 / fps]
    indices = [context - 1]
    step = 1.0 / sample_hz
    for time_s in np.arange(0.0, duration + step / 2, step):
        index = min(context + int(round(float(time_s) * fps)), last_generated)
        if index != indices[-1]:
            indices.append(index)
            times.append((index - context) / fps)
    if indices[-1] != last_generated:
        indices.append(last_generated)
        times.append(duration)
    return indices, times


def read_selected(path: str, wanted: list[int]) -> list[np.ndarray]:
    wanted_set = set(wanted)
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted_set:
                frames[index] = frame
                if len(frames) == len(wanted_set):
                    break
            index += 1
    finally:
        cap.release()
    missing = sorted(wanted_set - frames.keys())
    if missing:
        raise ValueError(f"missing frames {missing[:5]} from {path}")
    return [frames[index] for index in wanted]


def descriptor(orb: cv2.ORB, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    grey = cv2.cvtColor(cv2.resize(frame, SIZE), cv2.COLOR_BGR2GRAY)
    keypoints, desc = orb.detectAndCompute(grey, None)
    return np.float32([point.pt for point in keypoints]), desc


def verified_inliers(left: tuple[np.ndarray, np.ndarray | None],
                     right: tuple[np.ndarray, np.ndarray | None]) -> int:
    points_l, desc_l = left
    points_r, desc_r = right
    if desc_l is None or desc_r is None or len(points_l) < 8 or len(points_r) < 8:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_l, desc_r)
    if len(matches) < 8:
        return 0
    src = np.float32([points_l[item.queryIdx] for item in matches])
    dst = np.float32([points_r[item.trainIdx] for item in matches])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def score_video(row: dict[str, Any], *, local_prefix: str, remote_prefix: str,
                sample_hz: float, bank_s: float, cross_threshold: int,
                coherence_threshold: int, keypoint_threshold: int) -> dict[str, Any]:
    path = resolve_path(str(row["path"]), local_prefix, remote_prefix)
    context = int(row["context_frames"])
    fps = float(row["fps"])
    frame_count = int(row["decoded_frames"])
    indices, times = sample_plan(context, fps, frame_count, sample_hz)
    frames = read_selected(path, indices)
    orb = cv2.ORB_create(3000)
    desc = [descriptor(orb, frame) for frame in frames]
    keypoints = np.array([len(item[0]) for item in desc], dtype=int)
    cache: dict[tuple[int, int], int] = {}

    def match(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in cache:
            cache[key] = verified_inliers(desc[key[0]], desc[key[1]])
        return cache[key]

    radius = max(2, int(round(bank_s * sample_hz)))
    adjacent = np.array([match(i - 1, i) for i in range(1, len(desc))], dtype=int)
    bridge = np.zeros(len(desc), dtype=int)
    events: list[dict[str, Any]] = []
    # Index zero is the final real frame.  Require full banks on each side.
    for i in range(radius, len(desc) - radius):
        pre = list(range(i - radius, i))
        post = list(range(i, i + radius))
        bridge[i] = max(match(j, i) for j in pre)
        # A generous prescreen avoids computing bank cross-products at clearly
        # continuous boundaries.  It does not affect the final threshold.
        if bridge[i] >= max(cross_threshold * 2, coherence_threshold):
            continue
        cross = max(match(j, k) for j in pre for k in post)
        pre_coherence = float(np.median([match(j, j + 1) for j in pre[:-1]]))
        post_coherence = float(np.median([match(j, j + 1) for j in post[:-1]]))
        pre_keypoints = float(np.median(keypoints[pre]))
        post_keypoints = float(np.median(keypoints[post]))
        flagged = (cross < cross_threshold and
                   pre_coherence >= coherence_threshold and
                   post_coherence >= coherence_threshold and
                   pre_keypoints >= keypoint_threshold and
                   post_keypoints >= keypoint_threshold)
        if flagged:
            # Adjacent candidates describe the same replacement boundary.
            if events and times[i] - events[-1]["time_s"] <= bank_s:
                if cross < events[-1]["cross_inliers"]:
                    events[-1] = {
                        "time_s": times[i], "sample_index": i,
                        "frame_index": indices[i], "cross_inliers": int(cross),
                        "pre_coherence": pre_coherence,
                        "post_coherence": post_coherence,
                        "pre_keypoints": pre_keypoints,
                        "post_keypoints": post_keypoints,
                    }
            else:
                events.append({
                    "time_s": times[i], "sample_index": i,
                    "frame_index": indices[i], "cross_inliers": int(cross),
                    "pre_coherence": pre_coherence,
                    "post_coherence": post_coherence,
                    "pre_keypoints": pre_keypoints,
                    "post_keypoints": post_keypoints,
                })

    endpoint_rows = []
    for horizon in ENDPOINTS:
        occurred = [event for event in events if event["time_s"] <= horizon]
        endpoint_rows.append({
            "scene": row["scene"], "uid": row["uid"],
            "direction": row["direction"], "model": row["model"],
            "horizon_s": horizon,
            "long_relocation_flag": int(bool(occurred)),
            "first_relocation_s": occurred[0]["time_s"] if occurred else None,
            "relocation_events": len(occurred),
            "producer_version": VERSION,
        })
    return {
        "producer_version": VERSION,
        "scene": row["scene"], "model": row["model"],
        "path": path, "indices": indices, "times": times,
        "keypoints": keypoints.tolist(), "adjacent_inliers": adjacent.tolist(),
        "events": events, "endpoint_rows": endpoint_rows,
        "parameters": {
            "sample_hz": sample_hz, "bank_s": bank_s,
            "cross_threshold": cross_threshold,
            "coherence_threshold": coherence_threshold,
            "keypoint_threshold": keypoint_threshold,
        },
    }


def process_scene(rows: list[dict[str, Any]], destination: str, force: bool,
                  **kwargs: Any) -> tuple[str, int]:
    if not rows:
        raise ValueError("cannot process an empty scene")
    scene = rows[0]["scene"]
    if any(row["scene"] != scene for row in rows):
        raise ValueError("process_scene received rows from multiple scenes")
    models = [str(row["model"]) for row in rows]
    if len(models) != len(set(models)):
        raise ValueError(f"duplicate model rows for scene {scene}")
    output = Path(destination) / "scenes" / f"{scene}.json"
    parameters = {key: kwargs[key] for key in (
        "sample_hz", "bank_s", "cross_threshold",
        "coherence_threshold", "keypoint_threshold")}
    # Scene caches are resumability artefacts, not authoritative results.  A
    # matching producer version and parameter set is insufficient if a model
    # rollout was regenerated in place.  Record inexpensive filesystem
    # signatures for every input so such a replacement invalidates the whole
    # scene cache before any endpoint rows are reused.
    input_signatures = {}
    for row in rows:
        path = resolve_path(str(row["path"]), kwargs["local_prefix"],
                            kwargs["remote_prefix"])
        stat = os.stat(path)
        input_signatures[str(row["model"])] = {
            "path": path,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    if output.exists() and not force:
        prior = json.loads(output.read_text())
        if (prior.get("producer_version") == VERSION and
                prior.get("parameters") == parameters and
                prior.get("input_signatures") == input_signatures):
            return scene, len(prior["videos"])
    videos = [score_video(row, **kwargs) for row in rows]
    payload = {"producer_version": VERSION, "scene": scene,
               "parameters": parameters,
               "input_signatures": input_signatures,
               "videos": videos}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(output)
    return scene, len(videos)


def merge(destination: Path, expected_scenes: int | None,
          expected_models: int | None) -> None:
    files = sorted((destination / "scenes").glob("*.json"))
    if expected_scenes is not None and len(files) != expected_scenes:
        raise ValueError(f"expected {expected_scenes} scenes, found {len(files)}")
    rows, events, traces = [], [], []
    for path in files:
        payload = json.loads(path.read_text())
        if payload["producer_version"] != VERSION:
            raise ValueError(f"stale producer {path}")
        videos = payload["videos"]
        if len({video["model"] for video in videos}) != len(videos):
            raise ValueError(f"duplicate models in {path}")
        if expected_models is not None and len(videos) != expected_models:
            raise ValueError(
                f"expected {expected_models} models in {path}, found {len(videos)}")
        for video in videos:
            rows.extend(video["endpoint_rows"])
            for event in video["events"]:
                events.append({"scene": video["scene"], "model": video["model"], **event})
            for i, (frame, time_s, kp) in enumerate(zip(
                    video["indices"], video["times"], video["keypoints"])):
                traces.append({
                    "scene": video["scene"], "model": video["model"],
                    "sample_index": i, "frame_index": frame, "time_s": time_s,
                    "keypoints": kp,
                    "adjacent_inliers": (None if i == 0 else video["adjacent_inliers"][i - 1]),
                })
    frame = pd.DataFrame(rows)
    if len(frame):
        keys = ["model", "scene", "horizon_s"]
        if frame.duplicated(keys).any():
            raise ValueError("duplicate endpoint rows")
        if expected_scenes is not None and expected_models is not None:
            expected_rows = expected_scenes * expected_models * len(ENDPOINTS)
            if len(frame) != expected_rows:
                raise ValueError((len(frame), expected_rows))
        frame = frame.sort_values(keys)
    frame.to_csv(destination / "long_relocation_rows.csv", index=False)
    pd.DataFrame(events).to_csv(destination / "long_relocation_events.csv", index=False)
    pd.DataFrame(traces).to_csv(destination / "long_relocation_traces.csv", index=False)
    if len(frame):
        summary = frame.groupby(["model", "horizon_s"], as_index=False).agg(
            videos=("scene", "size"),
            relocation_rate=("long_relocation_flag", "mean"),
            median_first_relocation_s=("first_relocation_s", "median"),
        )
        summary["relocation_percent"] = 100 * summary.relocation_rate
        summary.to_csv(destination / "long_relocation_summary.csv", index=False)
    print(f"merged scenes={len(files)} rows={len(frame)} events={len(events)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-prefix", default="/home/ashish/ARRWM")
    parser.add_argument("--remote-prefix", default="")
    parser.add_argument("--sample-hz", type=float, default=4.0)
    parser.add_argument("--bank-s", type=float, default=1.0)
    parser.add_argument("--cross-threshold", type=int, default=20)
    parser.add_argument("--coherence-threshold", type=int, default=35)
    parser.add_argument("--keypoint-threshold", type=int, default=100)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scenes")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--expected-models", type=int)
    args = parser.parse_args()
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(destination, args.expected_scenes, args.expected_models)
        return
    if args.manifest is None:
        parser.error("--manifest is required unless --merge is used")
    manifest = pd.read_csv(args.manifest)
    scenes = sorted(manifest.scene.unique())
    if args.scenes:
        selected = set(args.scenes.split(","))
        scenes = [scene for scene in scenes if scene in selected]
    scenes = scenes[args.shard_index::args.shard_count]
    jobs = [manifest[manifest.scene.eq(scene)].to_dict("records") for scene in scenes]
    kwargs = {
        "destination": str(destination), "force": args.force,
        "local_prefix": args.local_prefix, "remote_prefix": args.remote_prefix,
        "sample_hz": args.sample_hz, "bank_s": args.bank_s,
        "cross_threshold": args.cross_threshold,
        "coherence_threshold": args.coherence_threshold,
        "keypoint_threshold": args.keypoint_threshold,
    }
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_scene, rows, **kwargs): rows[0]["scene"]
                   for rows in jobs}
        for future in as_completed(futures):
            scene, videos = future.result()
            completed += 1
            print(f"[{completed}/{len(jobs)}] {scene} videos={videos}", flush=True)
    print(f"shard complete {args.shard_index}/{args.shard_count}: {completed} scenes")


if __name__ == "__main__":
    main()
