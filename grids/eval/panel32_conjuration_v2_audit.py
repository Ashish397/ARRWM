#!/usr/bin/env python3
"""Fail-closed human audit queue for panel32 conjuration candidates.

This is deliberately a *separate* audit from the deployed conjuration metric.
It never writes ``eval_final/conjuration`` and it does not turn detector scores
into paper labels.  Instead it widens the tracked classes, retains every track
that passes the cheap birth/persistence/position gates, and renders evidence
for human adjudication even when the original materialisation score is <= 0.

The default windows are the five non-overlapping paper windows.  The detector
is run once on exactly the prefix needed for those windows (context through
30 seconds); the unreported overlapping 9--15 s window is not recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
QUALITY = ROOT / "code_release" / "evaluation" / "quality"
sys.path.insert(0, str(QUALITY))


WINDOW_SECONDS = 6
DEFAULT_WINDOWS = (0, 6, 12, 18, 24)

# Agents, vehicles, animals, and conspicuous carried/sports objects.  This is
# an audit recall set, not a claim that every newborn detection is conjured.
# In particular, ``person`` is intentionally present: excluding it caused the
# visible FrodoBots-u31/minWM false negative that motivated this audit.
SALIENT_MOBILE_CLASSES = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "suitcase", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
})


class AuditError(RuntimeError):
    """The audit input or output contract is incomplete or ambiguous."""


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    values = {item.strip() for item in value.split(",") if item.strip()}
    if not values:
        raise AuditError("empty comma-separated selection")
    return values


def parse_windows(value: str) -> tuple[int, ...]:
    try:
        windows = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise AuditError(f"windows must be integer seconds: {value!r}") from exc
    if not windows or len(set(windows)) != len(windows) or tuple(sorted(windows)) != windows:
        raise AuditError("windows must be non-empty, unique, and sorted")
    if windows[0] < 0 or windows[-1] + WINDOW_SECONDS > 30:
        raise AuditError("windows must lie within the 30-second generated horizon")
    if any(right - left < WINDOW_SECONDS for left, right in zip(windows, windows[1:])):
        raise AuditError("audit windows must not overlap")
    return windows


def load_work(
    manifest_path: Path,
    *,
    models: set[str] | None,
    scenes: set[str] | None,
    shard_index: int,
    shard_count: int,
) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise AuditError(f"missing video manifest: {manifest_path}")
    data = pd.read_csv(manifest_path)
    required = {
        "scene", "model", "path", "decoded_frames", "context_frames", "fps",
    }
    missing = required - set(data.columns)
    if missing:
        raise AuditError(f"video manifest missing columns: {sorted(missing)}")
    if data.duplicated(["scene", "model"]).any():
        raise AuditError("video manifest has duplicate (scene, model) rows")
    if data[list(required)].isna().any().any():
        raise AuditError("video manifest has null required fields")
    if models is not None:
        unknown = models - set(data.model.astype(str))
        if unknown:
            raise AuditError(f"unknown requested models: {sorted(unknown)}")
        data = data[data.model.astype(str).isin(models)]
    if scenes is not None:
        unknown = scenes - set(data.scene.astype(str))
        if unknown:
            raise AuditError(f"unknown requested scenes: {sorted(unknown)}")
        data = data[data.scene.astype(str).isin(scenes)]
    data = data.sort_values(["scene", "model"], kind="stable").reset_index(drop=True)
    if not 0 <= shard_index < shard_count:
        raise AuditError("invalid shard index/count")
    return data.iloc[shard_index::shard_count].reset_index(drop=True)


def needed_last_index(context_frames: int, fps: float, windows: Iterable[int]) -> int:
    end_s = max(windows) + WINDOW_SECONDS
    return int(context_frames) + int(round(float(end_s) * float(fps))) - 1


def decode_prefix(path: Path, last_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AuditError(f"cannot open video: {path}")
    frames = []
    for index in range(last_index + 1):
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise AuditError(f"missing frame {index} of required prefix: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def cheap_gate(track: dict[str, Any], n: int, width: int, height: int,
               ctx: int, detector: Any) -> tuple[bool, dict[str, Any]]:
    box = track["bbox"]
    area = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
    cx = detector.cen(box)[0] / width
    gates = {
        "salient_class": track["cls"] in SALIENT_MOBILE_CLASSES,
        "born_during_window": track["birth"] >= ctx,
        "peak_confidence": track["peak"] >= detector.HI,
        "minimum_birth_area": area >= detector.MIN_AREA,
        "interior_birth_centroid": detector.CXLO < cx < detector.CXHI,
        "not_lateral_edge_entry": detector.edge_trace(track, width) > detector.EDGE,
        "alive_in_tail": (track["last"] + 1) / n >= detector.PERSIST_TAIL,
        "persistent_since_birth": (
            track["hits"] / max(1, n - track["birth"]) >= detector.PERSIST_FRAC
        ),
    }
    return all(gates.values()), gates


def evidence_sheet(video: np.ndarray, candidate: dict[str, Any], *, fps: float,
                   title: str, output: Path) -> None:
    birth = int(candidate["global_birth_index"])
    x0, y0, x1, y1 = [int(round(v)) for v in candidate["box"]]
    height, width = video.shape[1:3]
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    offsets = (-1.0, -0.5, -0.125, 0.0, 0.25, 0.75, 2.0)
    cells = []
    for offset in offsets:
        index = min(len(video) - 1, max(0, birth + int(round(offset * fps))))
        frame = video[index].copy()
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 40, 40), 3)
        frame = cv2.resize(frame, (300, 173))
        cv2.putText(frame, f"{offset:+.3g}s  f={index}", (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cells.append(frame)
    body = np.concatenate(cells, axis=1)
    label = np.full((42, body.shape[1], 3), 18, np.uint8)
    cv2.putText(label, title[:220], (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1)
    cv2.putText(
        label,
        (f"score={candidate['legacy_score']:+.2f} onset={candidate['onset']} "
         f"zprobe={candidate['zprobe']} bncc={candidate['bncc']}  "
         "STATUS=UNADJUDICATED"),
        (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(
            np.concatenate([label, body], axis=0), cv2.COLOR_RGB2BGR)):
        raise AuditError(f"failed to write evidence sheet: {output}")


def candidate_rows(video: np.ndarray, detections: list[list[dict[str, Any]]],
                   *, context_frames: int, fps: float, windows: tuple[int, ...],
                   detector: Any) -> list[dict[str, Any]]:
    rows = []
    for start in windows:
        clip_start = 0 if start == 0 else (
            context_frames + int(round((start - 1) * fps))
        )
        analysis_ctx = context_frames if start == 0 else int(round(fps))
        clip_end = context_frames + int(round((start + WINDOW_SECONDS) * fps)) - 1
        part = video[clip_start:clip_end + 1]
        part_detections = detections[clip_start:clip_end + 1]
        tracks = detector.track(part_detections, part.shape[2], part.shape[1])
        candidates = []
        for track in tracks:
            passed, gates = cheap_gate(
                track, len(part), part.shape[2], part.shape[1], analysis_ctx, detector,
            )
            if not passed:
                continue
            features = detector.features(
                part, part_detections, track, len(part), analysis_ctx,
            )
            legacy_score = round(float(detector.score(features)), 2)
            features["legacy_score"] = legacy_score
            features["legacy_positive"] = bool(legacy_score > 0)
            features["cheap_gates"] = gates
            features["global_birth_index"] = clip_start + int(track["birth"])
            features["birth_s"] = round(
                (features["global_birth_index"] - context_frames) / fps, 3,
            )
            features["window_start_s"] = start
            features["window_end_s"] = start + WINDOW_SECONDS
            features["human_adjudication"] = "unadjudicated"
            candidates.append(features)
        candidates.sort(key=lambda row: (row["birth_s"], row["cls"], -row["peak"]))
        rows.append({
            "window_start_s": start,
            "window_end_s": start + WINDOW_SECONDS,
            "candidate_count": len(candidates),
            "human_adjudication_required": bool(candidates),
            "candidates": candidates,
        })
    return rows


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def flatten_record(record: dict[str, Any], record_path: Path) -> list[dict[str, Any]]:
    rows = []
    for window in record["windows"]:
        for index, candidate in enumerate(window["candidates"]):
            rows.append({
                "scene": record["scene"], "model": record["model"],
                "video_sha256": record["video_sha256"],
                "window_start_s": window["window_start_s"],
                "candidate_index": index, "cls": candidate["cls"],
                "birth_s": candidate["birth_s"], "peak": candidate["peak"],
                "legacy_score": candidate["legacy_score"],
                "legacy_positive": candidate["legacy_positive"],
                "human_adjudication": candidate["human_adjudication"],
                "record_path": str(record_path),
                "evidence_path": candidate.get("evidence_path", ""),
            })
    return rows


def reusable_record(target: Path, *, scene: str, model: str, digest: str,
                    windows: tuple[int, ...]) -> dict[str, Any] | None:
    if not target.is_file():
        return None
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("schema_version") != 1
        or record.get("audit") != "panel32_conjuration_v2"
        or record.get("scene") != scene
        or record.get("model") != model
        or record.get("video_sha256") != digest
        or record.get("salient_classes") != sorted(SALIENT_MOBILE_CLASSES)
        or [row.get("window_start_s") for row in record.get("windows", [])]
           != list(windows)
    ):
        return None
    for window in record["windows"]:
        for candidate in window.get("candidates", []):
            evidence = Path(str(candidate.get("evidence_path", "")))
            if not evidence.is_file() or evidence.stat().st_size == 0:
                return None
    return record


def validate_output_location(eval_root: Path, output: Path) -> None:
    root = eval_root.resolve()
    protected = {
        (eval_root / "conjuration").resolve(),
        (eval_root / "observations").resolve(),
        (eval_root / "final").resolve(),
    }
    resolved = output.resolve()
    if resolved == root or resolved in protected or any(
            protected_path in resolved.parents for protected_path in protected):
        raise AuditError(f"refusing to write audit into protected final output: {resolved}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path,
                        help="directory containing video_manifest.csv")
    parser.add_argument("--output", required=True, type=Path,
                        help="new, non-final audit directory")
    parser.add_argument("--models", help="comma-separated model subset")
    parser.add_argument("--scenes", help="comma-separated scene subset")
    parser.add_argument("--windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    windows = parse_windows(args.windows)
    eval_root = args.eval_root.resolve()
    output = args.output.resolve()
    validate_output_location(eval_root, output)
    work = load_work(
        eval_root / "video_manifest.csv",
        models=parse_csv_set(args.models), scenes=parse_csv_set(args.scenes),
        shard_index=args.shard_index, shard_count=args.shard_count,
    )
    required_frames = sum(
        needed_last_index(int(row.context_frames), float(row.fps), windows) + 1
        for row in work.itertuples()
    )
    plan = {
        "schema_version": 1, "videos": len(work), "windows": list(windows),
        "window_evaluations": len(work) * len(windows),
        "required_decoded_frames": int(required_frames),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "salient_classes": sorted(SALIENT_MOBILE_CLASSES),
        "output": str(output),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    # Imports/model construction are delayed so --plan-only works on CPU login nodes.
    import popin_backends as backends  # type: ignore
    import popin_detect as detector  # type: ignore

    detector.KEEP = set(detector.KEEP) | set(SALIENT_MOBILE_CLASSES)
    dense, crop = backends.build("rtdetr")
    detector.set_detector(crop)
    output.mkdir(parents=True, exist_ok=True)
    records_dir = output / "records"
    evidence_dir = output / "evidence"
    errors = []
    flat_rows = []
    for ordinal, row in enumerate(work.itertuples(), 1):
        target = records_dir / f"{row.scene}__{row.model}.json"
        try:
            path = Path(str(row.path)).resolve()
            if not path.is_file():
                raise AuditError(f"missing video: {path}")
            last = needed_last_index(int(row.context_frames), float(row.fps), windows)
            if last >= int(row.decoded_frames):
                raise AuditError(
                    f"required frame {last} outside decoded count {row.decoded_frames}: {path}"
                )
            digest = sha256_file(path)
            cached = reusable_record(
                target, scene=str(row.scene), model=str(row.model), digest=digest,
                windows=windows,
            )
            if cached is not None:
                flat_rows.extend(flatten_record(cached, target))
                count = sum(window["candidate_count"] for window in cached["windows"])
                print(f"[{ordinal}/{len(work)}] cached {row.scene} {row.model}: "
                      f"{count} candidates", flush=True)
                continue
            video = decode_prefix(path, last)
            if len(video) != last + 1:
                raise AuditError(f"decoded prefix length mismatch: {path}")
            detector.set_fps(float(row.fps))
            detections = dense(video)
            if len(detections) != len(video):
                raise AuditError(f"detector/frame length mismatch: {path}")
            window_rows = candidate_rows(
                video, detections, context_frames=int(row.context_frames),
                fps=float(row.fps), windows=windows, detector=detector,
            )
            record = {
                "schema_version": 1, "audit": "panel32_conjuration_v2",
                "label_status": "human_adjudication_required",
                "scene": str(row.scene), "model": str(row.model),
                "video_path": str(path), "video_sha256": digest,
                "fps": float(row.fps), "context_frames": int(row.context_frames),
                "decoded_prefix_frames": len(video), "windows": window_rows,
                "salient_classes": sorted(SALIENT_MOBILE_CLASSES),
            }
            for window in window_rows:
                for index, candidate in enumerate(window["candidates"]):
                    evidence = evidence_dir / (
                        f"{row.scene}__{row.model}__w{window['window_start_s']:02d}"
                        f"__c{index:03d}.png"
                    )
                    title = (
                        f"{row.model} {row.scene} w={window['window_start_s']}-"
                        f"{window['window_end_s']}s {candidate['cls']} "
                        f"birth={candidate['birth_s']:.3f}s"
                    )
                    evidence_sheet(video, candidate, fps=float(row.fps),
                                   title=title, output=evidence)
                    candidate["evidence_path"] = str(evidence)
            atomic_text(target, json.dumps(record, indent=2, sort_keys=True,
                                           default=json_default) + "\n")
            flat_rows.extend(flatten_record(record, target))
            count = sum(window["candidate_count"] for window in window_rows)
            print(f"[{ordinal}/{len(work)}] {row.scene} {row.model}: {count} candidates",
                  flush=True)
        except Exception as exc:  # fail after recording every shard error
            errors.append({"scene": str(row.scene), "model": str(row.model),
                           "error": repr(exc)})
            print(f"ERROR {row.scene} {row.model}: {exc}", flush=True)

    shard_tag = f"shard{args.shard_index:03d}-of-{args.shard_count:03d}"
    candidates_path = output / f"candidates_{shard_tag}.csv"
    pd.DataFrame(flat_rows, columns=[
        "scene", "model", "video_sha256", "window_start_s", "candidate_index",
        "cls", "birth_s", "peak", "legacy_score", "legacy_positive",
        "human_adjudication", "record_path", "evidence_path",
    ]).to_csv(candidates_path, index=False)
    summary = {
        **plan, "processed_videos": len(work) - len(errors),
        "candidate_count": len(flat_rows), "errors": errors,
        "status": "pass" if not errors else "fail",
        "candidate_csv": str(candidates_path),
    }
    atomic_text(output / f"summary_{shard_tag}.json",
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if errors:
        raise AuditError(f"audit failed on {len(errors)} videos")


if __name__ == "__main__":
    main()
