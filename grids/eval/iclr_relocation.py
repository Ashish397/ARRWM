"""Family-balanced ORB/RANSAC relocation evaluation for ICLR rollouts.

This ports the AAAI Reloc-Cal-v2 instrument to the ICLR model panel.  Each
candidate endpoint is matched against four other-family model endpoints and
against real frames from the conditioning span that candidate actually saw.
The score is the largest verified-inlier count across those eligible
references.  The AAAI decision is score < 50 at the six-second anchor; later
endpoints retain the same raw score and flag as a long-horizon extension.

The producer is scene-sharded and resumable.  It writes one JSON per scene so
an interrupted holder command can be fixed and resumed without repeating
completed work.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


VERSION = "iclr-reloc-v1-aaai-orb3000-ransac5-family5"
ENDPOINTS = (6, 12, 18, 24, 30)
SEAT = {
    "lingbot": "lingbot",
    "dreamx": "dreamx",
    "minwm": "minwm",
    "matrixgame2": "matrixgame2",
    "ours": "ours_recovery_base",
}


def family(model: str) -> str:
    if model.startswith("ours_"):
        return "ours"
    if model.startswith("minwm"):
        return "minwm"
    return model


def real_indices(model: str, context_frames: int) -> tuple[int, ...]:
    if context_frames <= 1:
        return (0,)
    if model == "minwm" and context_frames == 29:
        return (16, 20, 24, 28)
    if model == "minwm" and context_frames == 13:
        return (0, 4, 8, 12)
    if model == "minwm_ode" or model.startswith("ours_"):
        return (20, 24, 28, 32)
    # Fallback spans the actual real context without looking beyond it.
    return tuple(np.linspace(0, context_frames - 1, 4).round().astype(int))


def endpoint_index(context_frames: int, fps: float, frame_count: int,
                   horizon: int) -> int:
    # Preserve the deployed AAAI six-second anchor.  Later endpoints use the
    # last frame contained within H seconds of generated video.
    index = (context_frames + int(round(horizon * fps)) if horizon == 6
             else context_frames + int(round(horizon * fps)) - 1)
    if index >= frame_count:
        raise IndexError((context_frames, fps, frame_count, horizon, index))
    return index


def resolve_path(path: str, local_prefix: str, remote_prefix: str) -> str:
    if os.path.exists(path):
        return path
    if local_prefix and path.startswith(local_prefix):
        candidate = remote_prefix + path[len(local_prefix):]
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(path)


def read_frames(path: str, indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open {path}")
    frames: dict[int, np.ndarray] = {}
    try:
        for index in sorted(set(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"undecodable frame {index}: {path}")
            frames[index] = frame
    finally:
        cap.release()
    return frames


def descriptor(orb: cv2.ORB, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    grey = cv2.cvtColor(cv2.resize(image, (640, 352)), cv2.COLOR_BGR2GRAY)
    keypoints, desc = orb.detectAndCompute(grey, None)
    points = np.float32([kp.pt for kp in keypoints])
    return points, desc


def inliers(left: tuple[np.ndarray, np.ndarray | None],
            right: tuple[np.ndarray, np.ndarray | None]) -> int:
    points_l, desc_l = left
    points_r, desc_r = right
    if desc_l is None or desc_r is None or len(points_l) < 8 or len(points_r) < 8:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_l, desc_r)
    if len(matches) < 8:
        return 0
    src = np.float32([points_l[m.queryIdx] for m in matches])
    dst = np.float32([points_r[m.trainIdx] for m in matches])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def signature(path: str) -> dict[str, Any]:
    stat = os.stat(path)
    return {"path": path, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def process_scene(rows: list[dict[str, Any]], seed_dir: str, destination: str,
                  local_prefix: str, remote_prefix: str, force: bool) -> tuple[str, int, int]:
    scene = rows[0]["scene"]
    uid = rows[0]["uid"]
    output = Path(destination) / "scenes" / f"{scene}.json"
    paths = {
        row["model"]: resolve_path(str(row["path"]), local_prefix, remote_prefix)
        for row in rows
    }
    input_signatures = {model: signature(path) for model, path in paths.items()}
    seed_path = str(Path(seed_dir) / f"seed65_{uid}.mp4")
    seed_signature = signature(seed_path)
    if output.exists() and not force:
        try:
            prior = json.loads(output.read_text())
            if (prior.get("producer_version") == VERSION and
                    prior.get("input_signatures") == input_signatures and
                    prior.get("seed_signature") == seed_signature):
                return scene, len(prior["rows"]), len(prior["pairs"])
        except Exception:
            pass

    expected_models = set(SEAT.values()) | {row["model"] for row in rows}
    if not set(SEAT.values()).issubset(paths):
        raise ValueError(f"{scene}: incomplete family panel")
    if len(rows) != 15 or len(paths) != 15:
        raise ValueError(f"{scene}: expected 15 models, got {len(rows)}")

    orb = cv2.ORB_create(3000)
    endpoint_desc: dict[tuple[str, int], tuple[np.ndarray, np.ndarray | None]] = {}
    endpoint_meta: dict[tuple[str, int], tuple[int, float]] = {}
    for row in rows:
        model = row["model"]
        context = int(row["context_frames"])
        fps = float(row["fps"])
        count = int(row["decoded_frames"])
        indices = {h: endpoint_index(context, fps, count, h) for h in ENDPOINTS}
        frames = read_frames(paths[model], list(indices.values()))
        for horizon, index in indices.items():
            endpoint_desc[(model, horizon)] = descriptor(orb, frames[index])
            endpoint_meta[(model, horizon)] = (index, (index - context) / fps)

    needed_real = sorted({i for row in rows for i in real_indices(
        row["model"], int(row["context_frames"]))})
    real_frames = read_frames(seed_path, needed_real)
    real_desc = {i: descriptor(orb, real_frames[i]) for i in needed_real}

    result_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for row in rows:
        model = row["model"]
        fam = family(model)
        seats = dict(SEAT)
        seats[fam] = model
        peer_models = sorted(m for m in seats.values() if m != model)
        refs = real_indices(model, int(row["context_frames"]))
        for horizon in ENDPOINTS:
            candidate = endpoint_desc[(model, horizon)]
            evidence: list[tuple[str, str, int]] = []
            for peer in peer_models:
                evidence.append((peer, "model", inliers(candidate, endpoint_desc[(peer, horizon)])))
            for index in refs:
                evidence.append((f"real_{index}", "real", inliers(candidate, real_desc[index])))
            winner, winner_type, score = max(evidence, key=lambda item: item[2])
            peer_best = max(value for _, kind, value in evidence if kind == "model")
            real_best = max(value for _, kind, value in evidence if kind == "real")
            index, timestamp = endpoint_meta[(model, horizon)]
            result_rows.append({
                "scene": scene, "uid": uid, "direction": row["direction"],
                "model": model, "family": fam, "horizon_s": horizon,
                "endpoint_index": index, "endpoint_timestamp_s": round(timestamp, 5),
                "endpoint_keypoints": int(len(candidate[0])),
                "panel_inliers": int(score), "peer_best_inliers": int(peer_best),
                "real_best_inliers": int(real_best), "best_reference": winner,
                "best_reference_type": winner_type,
                "relocation_flag_50": int(score < 50),
                "reference_indices": ",".join(map(str, refs)),
                "peer_models": ",".join(peer_models),
                "panel_size": len(evidence),
                "threshold_status": ("AAAI Reloc-Cal-v2 cutoff transferred to five-family ICLR panel"
                                     if horizon == 6 else
                                     "long-horizon extension; requires visual validation"),
                "producer_version": VERSION,
            })
            for reference, kind, value in evidence:
                pair_rows.append({
                    "scene": scene, "model": model, "horizon_s": horizon,
                    "reference": reference, "reference_type": kind,
                    "inliers": int(value), "producer_version": VERSION,
                })

    payload = {
        "producer_version": VERSION,
        "scene": scene,
        "input_signatures": input_signatures,
        "seed_signature": seed_signature,
        "rows": result_rows,
        "pairs": pair_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload))
    temporary.replace(output)
    return scene, len(result_rows), len(pair_rows)


def merge(destination: Path, expected_scenes: int | None = None) -> None:
    rows, pairs = [], []
    files = sorted((destination / "scenes").glob("*.json"))
    if expected_scenes is not None and len(files) != expected_scenes:
        raise ValueError(f"expected {expected_scenes} scenes, found {len(files)}")
    for path in files:
        data = json.loads(path.read_text())
        if data["producer_version"] != VERSION:
            raise ValueError(f"stale producer: {path}")
        rows.extend(data["rows"])
        pairs.extend(data["pairs"])
    frame = pd.DataFrame(rows).sort_values(["model", "scene", "horizon_s"])
    pair_frame = pd.DataFrame(pairs).sort_values(
        ["model", "scene", "horizon_s", "reference_type", "reference"])
    if len(frame):
        if frame.duplicated(["model", "scene", "horizon_s"]).any():
            raise ValueError("duplicate relocation rows")
        if expected_scenes is not None and len(frame) != expected_scenes * 15 * 5:
            raise ValueError((len(frame), expected_scenes * 15 * 5))
    frame.to_csv(destination / "relocation_rows.csv", index=False)
    pair_frame.to_csv(destination / "relocation_pair_evidence.csv", index=False)
    if len(frame):
        summary = frame.groupby(["model", "horizon_s"], as_index=False).agg(
            videos=("scene", "size"),
            relocation_rate=("relocation_flag_50", "mean"),
            median_panel_inliers=("panel_inliers", "median"),
            median_peer_inliers=("peer_best_inliers", "median"),
            median_real_inliers=("real_best_inliers", "median"),
        )
        summary["relocation_percent"] = 100 * summary.relocation_rate
        summary.to_csv(destination / "relocation_summary.csv", index=False)
    print(f"merged scenes={len(files)} rows={len(frame)} pairs={len(pair_frame)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path)
    parser.add_argument("--local-prefix", default="/home/ashish/ARRWM")
    parser.add_argument("--remote-prefix", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scenes", help="optional comma-separated scene keys")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--expected-scenes", type=int)
    args = parser.parse_args()
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(destination, args.expected_scenes)
        return
    if args.manifest is None or args.seed_dir is None:
        parser.error("--manifest and --seed-dir are required unless --merge is used")
    manifest = pd.read_csv(args.manifest)
    required = {"scene", "uid", "direction", "model", "path", "fps",
                "context_frames", "decoded_frames"}
    if not required.issubset(manifest.columns):
        raise ValueError(required - set(manifest.columns))
    scenes = sorted(manifest.scene.unique())
    if args.scenes:
        selected = set(args.scenes.split(","))
        scenes = [scene for scene in scenes if scene in selected]
    scenes = scenes[args.shard_index::args.shard_count]
    jobs = [manifest[manifest.scene == scene].to_dict("records") for scene in scenes]
    kwargs = dict(seed_dir=str(args.seed_dir), destination=str(destination),
                  local_prefix=args.local_prefix, remote_prefix=args.remote_prefix,
                  force=args.force)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_scene, rows, **kwargs): rows[0]["scene"]
                   for rows in jobs}
        for future in as_completed(futures):
            scene, nrows, npairs = future.result()
            completed += 1
            print(f"[{completed}/{len(jobs)}] {scene} rows={nrows} pairs={npairs}", flush=True)
    print(f"shard complete {args.shard_index}/{args.shard_count}: {completed} scenes")


if __name__ == "__main__":
    main()
