#!/usr/bin/env python3
"""Merge, review, and finalize the panel32 conjuration-v2 human audit.

The authoritative producer is ``panel32_conjuration_v2_audit.py``. Its
per-video records contain five ``windows`` and zero or more high-recall
``candidates`` with already-rendered evidence sheets. This program never
re-runs the detector and never silently drops proposals:

* ``merge`` validates one record for every manifest (scene, model), all five
  windows, and every evidence file, then writes a stable adjudication CSV;
* ``render`` indexes the existing evidence and, when the videos are available,
  adds a longer-history strip reaching three seconds before birth;
* ``finalize`` accepts output only when every unchanged candidate has a 0/1
  verdict and a non-empty reason, then emits the complete video-window grid.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd


WINDOWS = (0, 6, 12, 18, 24)
WINDOW_SECONDS = 6
AUDIT_NAME = "panel32_conjuration_v2"
MERGE_SCHEMA_VERSION = 1
CANDIDATE_COLUMNS = (
    "candidate_id", "scene", "model", "source_window_start_s", "birth_s",
    "global_birth_index", "class", "legacy_score", "legacy_positive", "peak",
    "area", "onset", "zprobe", "bncc", "box", "video_sha256",
    "evidence_path", "record_path", "adjudication", "reason",
)
IMMUTABLE_COLUMNS = CANDIDATE_COLUMNS[:-2]


class ReviewError(RuntimeError):
    """The audit records or human review are incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def merge_sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".merge.json")


def manifest_table(path: Path) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    if not path.is_file():
        raise ReviewError(f"missing manifest: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "scene", "model", "path", "fps", "context_frames", "decoded_frames",
        "width", "height",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ReviewError(f"manifest missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any() or any(
        frame[column].astype(str).str.strip().eq("").any() for column in required
    ):
        raise ReviewError("manifest contains null/empty required fields")
    for column in ("fps", "context_frames", "decoded_frames", "width", "height"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or ~np.isfinite(numeric).all() or (numeric <= 0).any():
            raise ReviewError(f"manifest contains invalid {column}")
        if column != "fps" and not np.equal(numeric, np.floor(numeric)).all():
            raise ReviewError(f"manifest contains non-integral {column}")
    if frame.duplicated(["scene", "model"]).any():
        raise ReviewError("manifest contains duplicate (scene, model) keys")
    keys = set(zip(frame.scene.astype(str), frame.model.astype(str)))
    if len(keys) != len(frame):
        raise ReviewError("manifest key count mismatch")
    return frame, keys


def records_directory(path: Path) -> Path:
    candidate = path / "records"
    return candidate if candidate.is_dir() else path


def finite_number(value: Any, where: str) -> float:
    try:
        answer = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewError(f"{where}: expected numeric value, got {value!r}") from exc
    if not math.isfinite(answer):
        raise ReviewError(f"{where}: non-finite value")
    return answer


def canonical_box(value: Any, *, width: int, height: int, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ReviewError(f"{where}: box must contain four values")
    box = [finite_number(item, where) for item in value]
    x0, y0, x1, y1 = box
    # Detector boxes are rounded to one decimal in the producer. Backends can
    # return a coordinate just outside the raster (the largest overshoot in
    # the complete 4,320-video audit is 1.3 px), even though
    # drawing/measurement clips it to the image. Canonicalize only this narrow
    # boundary-rounding range; anything beyond 1.5 px remains a hard failure.
    tolerance = 1.5
    if not (-tolerance <= x0 < x1 <= width + tolerance and
            -tolerance <= y0 < y1 <= height + tolerance):
        raise ReviewError(f"{where}: invalid/out-of-frame box {box}")
    clipped = [max(0.0, x0), max(0.0, y0), min(float(width), x1),
               min(float(height), y1)]
    if not (clipped[0] < clipped[2] and clipped[1] < clipped[3]):
        raise ReviewError(f"{where}: empty box after boundary clipping {box}")
    return [round(item, 1) for item in clipped]


def candidate_id(row: dict[str, Any]) -> str:
    # Window and bbox are both included. Two same-class objects born at the
    # same time remain distinct instead of being time/class-deduplicated.
    material = "|".join([
        str(row["scene"]), str(row["model"]),
        str(int(row["source_window_start_s"])), str(row["class"]),
        f"{float(row['birth_s']):.3f}", str(int(row["global_birth_index"])),
        ",".join(f"{float(value):.1f}" for value in json.loads(str(row["box"]))),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def immutable_digest(row: dict[str, Any]) -> str:
    # Canonicalize values so the digest survives a CSV write/read round trip.
    canonical: dict[str, Any] = {}
    integer_columns = {"source_window_start_s", "global_birth_index", "legacy_positive"}
    float_columns = {"birth_s", "legacy_score", "peak", "area", "onset", "zprobe", "bncc"}
    for column in IMMUTABLE_COLUMNS:
        value = row[column]
        if column in integer_columns:
            canonical[column] = int(value)
        elif column in float_columns:
            canonical[column] = float(value)
        elif column == "box":
            canonical[column] = [float(item) for item in json.loads(str(value))]
        else:
            canonical[column] = str(value)
    material = json.dumps(
        canonical,
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_record(
    path: Path,
    payload: dict[str, Any],
    manifest_row: Any,
    evidence_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # ``merge`` indexes the manifest by (scene, model), so pandas removes
    # those columns from the selected Series and retains them in ``name``.
    # Accept a record-like object too, which keeps this validator convenient
    # for direct tests and callers outside ``merge``.
    if hasattr(manifest_row, "scene") and hasattr(manifest_row, "model"):
        scene, model = str(manifest_row.scene), str(manifest_row.model)
    else:
        name = getattr(manifest_row, "name", None)
        if not isinstance(name, tuple) or len(name) != 2:
            raise ReviewError("manifest row has no (scene, model) identity")
        scene, model = map(str, name)
    where = f"{scene}/{model}"
    if payload.get("schema_version") != 1 or payload.get("audit") != AUDIT_NAME:
        raise ReviewError(f"{where}: wrong audit schema/profile in {path}")
    if (str(payload.get("scene")), str(payload.get("model"))) != (scene, model):
        raise ReviewError(f"{where}: record identity mismatch in {path}")
    if Path(str(payload.get("video_path", ""))).resolve() != Path(str(manifest_row.path)).resolve():
        raise ReviewError(f"{where}: video path differs from manifest")
    digest = str(payload.get("video_sha256", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ReviewError(f"{where}: invalid video SHA256")
    fps = finite_number(payload.get("fps"), f"{where}.fps")
    if not math.isclose(fps, float(manifest_row.fps), abs_tol=1e-6):
        raise ReviewError(f"{where}: fps differs from manifest")
    context = int(payload.get("context_frames", -1))
    if context != int(manifest_row.context_frames):
        raise ReviewError(f"{where}: context_frames differs from manifest")
    expected_prefix = context + int(round(30 * fps))
    if int(payload.get("decoded_prefix_frames", -1)) != expected_prefix:
        raise ReviewError(f"{where}: decoded prefix does not end at 30 seconds")
    windows = payload.get("windows")
    if not isinstance(windows, list) or [w.get("window_start_s") for w in windows] != list(WINDOWS):
        raise ReviewError(f"{where}: expected exactly the five ordered windows {WINDOWS}")

    candidates: list[dict[str, Any]] = []
    for window in windows:
        start = int(window["window_start_s"])
        if int(window.get("window_end_s", -1)) != start + WINDOW_SECONDS:
            raise ReviewError(f"{where}/w{start}: invalid window end")
        values = window.get("candidates")
        if not isinstance(values, list):
            raise ReviewError(f"{where}/w{start}: candidates must be a list")
        if int(window.get("candidate_count", -1)) != len(values):
            raise ReviewError(f"{where}/w{start}: candidate_count mismatch")
        if bool(window.get("human_adjudication_required")) != bool(values):
            raise ReviewError(f"{where}/w{start}: adjudication-required mismatch")
        for ordinal, event in enumerate(values):
            cwhere = f"{where}/w{start}/c{ordinal}"
            if not isinstance(event, dict):
                raise ReviewError(f"{cwhere}: candidate is not an object")
            if (int(event.get("window_start_s", -1)) != start or
                    int(event.get("window_end_s", -1)) != start + WINDOW_SECONDS):
                raise ReviewError(f"{cwhere}: candidate/window mismatch")
            birth_s = finite_number(event.get("birth_s"), f"{cwhere}.birth_s")
            if not start <= birth_s < start + WINDOW_SECONDS:
                raise ReviewError(f"{cwhere}: birth outside source window")
            global_birth = int(event.get("global_birth_index", -1))
            expected_birth_s = (global_birth - context) / fps
            if not math.isclose(birth_s, expected_birth_s, abs_tol=0.0011):
                raise ReviewError(f"{cwhere}: global birth index/time mismatch")
            box = canonical_box(event.get("box"), width=int(manifest_row.width),
                                height=int(manifest_row.height), where=cwhere)
            gates = event.get("cheap_gates")
            if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
                raise ReviewError(f"{cwhere}: candidate did not pass every cheap gate")
            evidence = Path(str(event.get("evidence_path", ""))).resolve()
            if evidence_root != evidence and evidence_root not in evidence.parents:
                raise ReviewError(f"{cwhere}: evidence path escapes audit evidence directory")
            if not evidence.is_file() or evidence.stat().st_size == 0:
                raise ReviewError(f"{cwhere}: missing/empty evidence {evidence}")
            legacy_positive = event.get("legacy_positive")
            if not isinstance(legacy_positive, bool):
                raise ReviewError(f"{cwhere}: legacy_positive must be boolean")
            row = {
                "scene": scene, "model": model, "source_window_start_s": start,
                "birth_s": round(birth_s, 3), "global_birth_index": global_birth,
                "class": str(event.get("cls", "")),
                "legacy_score": finite_number(event.get("legacy_score"), f"{cwhere}.legacy_score"),
                "legacy_positive": int(legacy_positive),
                "peak": finite_number(event.get("peak"), f"{cwhere}.peak"),
                "area": finite_number(event.get("area"), f"{cwhere}.area"),
                "onset": finite_number(event.get("onset"), f"{cwhere}.onset"),
                "zprobe": finite_number(event.get("zprobe"), f"{cwhere}.zprobe"),
                "bncc": finite_number(event.get("bncc"), f"{cwhere}.bncc"),
                "box": json.dumps(box, separators=(",", ":")),
                "video_sha256": digest, "evidence_path": str(evidence),
                "record_path": str(path.resolve()),
            }
            if not row["class"]:
                raise ReviewError(f"{cwhere}: empty class")
            row["candidate_id"] = candidate_id(row)
            row["adjudication"] = ""
            row["reason"] = ""
            candidates.append(row)
    metadata = {"scene": scene, "model": model, "video_sha256": digest,
                "record_path": str(path.resolve()), "record_sha256": sha256_file(path)}
    return candidates, metadata


def _preserve_reviews(output: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if not output.exists():
        return frame
    previous = pd.read_csv(output, keep_default_na=False, dtype=str)
    if set(previous.columns) != set(CANDIDATE_COLUMNS):
        raise ReviewError(f"refusing to overwrite incompatible review CSV: {output}")
    if previous.candidate_id.duplicated().any():
        raise ReviewError(f"refusing to read duplicate candidate IDs from {output}")
    if set(previous.candidate_id) != set(frame.candidate_id.astype(str)):
        raise ReviewError("refusing to overwrite review CSV with a changed candidate set")
    reviews = previous.set_index("candidate_id")[["adjudication", "reason"]]
    frame = frame.copy()
    frame["adjudication"] = frame.candidate_id.map(reviews.adjudication)
    frame["reason"] = frame.candidate_id.map(reviews.reason)
    return frame


def merge(args: argparse.Namespace) -> None:
    manifest, keys = manifest_table(args.manifest)
    records_dir = records_directory(args.candidates.resolve())
    audit_root = records_dir.parent
    evidence_root = (audit_root / "evidence").resolve()
    paths = sorted(records_dir.glob("*.json"))
    if len(paths) != len(keys):
        raise ReviewError(f"expected {len(keys)} record JSONs, found {len(paths)}")
    by_key = manifest.set_index(["scene", "model"], verify_integrity=True)
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    record_metadata: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewError(f"cannot read record {path}: {exc}") from exc
        key = (str(payload.get("scene")), str(payload.get("model")))
        if key not in keys or key in seen:
            raise ReviewError(f"unexpected or duplicate record key: {key}")
        expected_name = f"{key[0]}__{key[1]}.json"
        if path.name != expected_name:
            raise ReviewError(f"record filename/identity mismatch: {path.name} != {expected_name}")
        candidates, metadata = _validate_record(
            path, payload, by_key.loc[key], evidence_root,
        )
        rows.extend(candidates)
        record_metadata.append(metadata)
        seen.add(key)
    if seen != keys:
        raise ReviewError(f"missing records for {len(keys - seen)} manifest keys")

    frame = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(
            ["model", "scene", "source_window_start_s", "birth_s", "class", "candidate_id"],
            kind="stable",
        ).reset_index(drop=True)
        if not frame.candidate_id.is_unique:
            duplicates = frame.loc[frame.candidate_id.duplicated(False), "candidate_id"].tolist()
            raise ReviewError(f"candidate ID collision: {duplicates[:8]}")
    frame = _preserve_reviews(args.output, frame)
    immutable = {
        str(row["candidate_id"]): immutable_digest(row)
        for row in frame.to_dict("records")
    }
    sidecar = {
        "schema_version": MERGE_SCHEMA_VERSION, "audit": AUDIT_NAME,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "manifest_video_keys": len(keys), "windows": list(WINDOWS),
        "window_rows_expected": len(keys) * len(WINDOWS),
        "candidate_count": len(frame), "candidate_ids": frame.candidate_id.tolist(),
        "immutable_candidate_sha256": immutable,
        "records": sorted(record_metadata, key=lambda row: (row["scene"], row["model"])),
    }
    atomic_csv(frame[list(CANDIDATE_COLUMNS)], args.output)
    atomic_text(merge_sidecar_path(args.output),
                json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "record_jsons": len(paths), "validated_windows": len(keys) * len(WINDOWS),
        "candidates": len(frame), "output": str(args.output),
        "merge_sidecar": str(merge_sidecar_path(args.output)),
    }, indent=2))


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = set(CANDIDATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ReviewError(f"candidate CSV missing columns: {sorted(missing)}")
    if frame.candidate_id.astype(str).duplicated().any():
        raise ReviewError("candidate IDs are not unique")
    return frame[list(CANDIDATE_COLUMNS)].copy()


def _caption(row: Mapping[str, Any], ordinal: int, total: int) -> str:
    return (
        f"{ordinal}/{total}  {row['candidate_id']}  {row['model']}/{row['scene']}  "
        f"w={int(row['source_window_start_s'])}-{int(row['source_window_start_s'])+6}s  "
        f"class={row['class']} birth={float(row['birth_s']):.3f}s  "
        f"legacy={float(row['legacy_score']):+.2f}"
    )


def _read_frames(path: str, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = sorted(set(max(0, int(index)) for index in indices))
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ReviewError(f"cannot open {path}")
    frames: dict[int, np.ndarray] = {}
    cursor = 0
    for index in range(wanted[-1] + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if index == wanted[cursor]:
            frames[index] = frame
            cursor += 1
            if cursor == len(wanted):
                break
    capture.release()
    if set(frames) != set(wanted):
        raise ReviewError(f"missing requested frames from {path}")
    return frames


def _enriched_strip(row: Mapping[str, Any], source: Any, output: Path) -> None:
    fps, context, total = float(source.fps), int(source.context_frames), int(source.decoded_frames)
    birth = int(row["global_birth_index"])
    offsets = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    indices = [context - 1] + [
        min(total - 1, max(context, birth + int(round(offset * fps))))
        for offset in offsets
    ]
    frames = _read_frames(str(source.path), indices)
    box = [int(round(value)) for value in json.loads(str(row["box"]))]
    labels = ["conditioning"] + [f"birth {offset:+.1f}s" for offset in offsets]
    cells = []
    for position, (index, label) in enumerate(zip(indices, labels)):
        frame = frames[index].copy()
        if position == 5:  # conditioning plus four negative offsets
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
        frame = cv2.resize(frame, (300, 173), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (299, 23), (15, 15, 15), -1)
        cv2.putText(frame, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(frame)
    strip = np.concatenate(cells, axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), strip, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise ReviewError(f"failed to write enriched evidence {output}")


def _contact_pages(candidates: pd.DataFrame, output: Path, per_page: int = 12) -> list[Path]:
    pages: list[Path] = []
    for page_index, offset in enumerate(range(0, len(candidates), per_page), 1):
        entries = candidates.iloc[offset:offset + per_page].to_dict("records")
        strips = []
        for ordinal, row in enumerate(entries, offset + 1):
            display_path = Path(str(row["review_evidence_path"]))
            image = cv2.imread(str(display_path))
            if image is None:
                raise ReviewError(f"cannot read evidence {display_path}")
            target_width = 1600
            target_height = max(1, round(image.shape[0] * target_width / image.shape[1]))
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            header = np.full((34, target_width, 3), 20, np.uint8)
            cv2.putText(header, _caption(row, ordinal, len(candidates))[:210], (6, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                        cv2.LINE_AA)
            strips.append(np.concatenate([header, image], axis=0))
        if not strips:
            continue
        page = output / f"contact_{page_index:04d}.jpg"
        if not cv2.imwrite(str(page), np.concatenate(strips, axis=0),
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ReviewError(f"failed to write contact page {page}")
        pages.append(page)
    return pages


def render(args: argparse.Namespace) -> None:
    candidates = _load_candidates(args.candidates)
    manifest, keys = manifest_table(args.manifest)
    index = manifest.set_index(["scene", "model"], verify_integrity=True)
    args.output.mkdir(parents=True, exist_ok=True)
    enriched_dir = args.output / "enriched"
    review_paths = []
    html_rows = []
    for ordinal, row in enumerate(candidates.to_dict("records"), 1):
        key = (str(row["scene"]), str(row["model"]))
        if key not in keys:
            raise ReviewError(f"candidate absent from manifest: {key}")
        existing = Path(str(row["evidence_path"])).resolve()
        if not existing.is_file() or existing.stat().st_size == 0:
            raise ReviewError(f"missing/empty existing evidence: {existing}")
        source = index.loc[key]
        video_path = Path(str(source.path)).resolve()
        enriched = enriched_dir / f"{ordinal:05d}_{row['candidate_id']}.jpg"
        if video_path.is_file():
            _enriched_strip(row, source, enriched)
            review_path = enriched
        else:
            # The authoritative, already-rendered strip remains reviewable;
            # expose the full source path so ambiguous cases can be revisited.
            review_path = existing
        review_paths.append(str(review_path))
        caption = _caption(row, ordinal, len(candidates))
        existing_rel = os.path.relpath(existing, args.output.resolve())
        review_rel = os.path.relpath(review_path, args.output.resolve())
        source_note = (f"<a href='{html.escape(os.path.relpath(video_path, args.output.resolve()), quote=True)}'>"
                       "Full source video</a>" if video_path.is_file()
                       else f"Full source video: {html.escape(str(video_path))}")
        html_rows.append(
            "<article>" f"<h3>{html.escape(caption)}</h3>"
            f"<p><a href='{html.escape(existing_rel, quote=True)}'>Original audit strip</a> | "
            f"{source_note}</p>"
            f"<img loading='lazy' src='{html.escape(review_rel, quote=True)}' "
            f"alt='{html.escape(caption, quote=True)}'>"
            f"<p>adjudication={html.escape(str(row['adjudication']))}; "
            f"reason={html.escape(str(row['reason']))}</p></article>"
        )
    contact_frame = candidates.copy()
    contact_frame["review_evidence_path"] = review_paths
    pages = _contact_pages(contact_frame, args.output)
    contacts = "".join(
        f"<li><a href='{html.escape(page.name)}'>{html.escape(page.name)}</a></li>"
        for page in pages
    )
    page = """<!doctype html><meta charset="utf-8">
<title>Panel32 conjuration-v2 review</title>
<style>body{font:14px sans-serif;margin:24px}article{margin:0 0 28px}img{max-width:100%;height:auto}h3{font-size:14px}</style>
""" + f"<h1>Panel32 conjuration-v2 candidates ({len(candidates)})</h1>" \
        + f"<h2>Contact sheets</h2><ul>{contacts}</ul>" + "\n".join(html_rows)
    atomic_text(args.output / "index.html", page)
    print(json.dumps({"candidates": len(candidates), "contact_pages": len(pages),
                      "index": str(args.output / "index.html")}, indent=2))


def _validate_adjudication(
    candidates: pd.DataFrame, sidecar: dict[str, Any], manifest_sha: str,
) -> pd.DataFrame:
    if (sidecar.get("schema_version") != MERGE_SCHEMA_VERSION or
            sidecar.get("audit") != AUDIT_NAME or
            sidecar.get("manifest_sha256") != manifest_sha or
            sidecar.get("windows") != list(WINDOWS)):
        raise ReviewError("merge sidecar does not match manifest/audit schema")
    ids = candidates.candidate_id.astype(str).tolist()
    if ids != sidecar.get("candidate_ids") or len(ids) != int(sidecar.get("candidate_count", -1)):
        raise ReviewError("adjudication candidate IDs/order differ from merge sidecar")
    expected_digests = sidecar.get("immutable_candidate_sha256", {})
    for row in candidates.to_dict("records"):
        candidate = str(row["candidate_id"])
        if candidate_id(row) != candidate:
            raise ReviewError(f"candidate ID fields were changed: {candidate}")
        if immutable_digest(row) != expected_digests.get(candidate):
            raise ReviewError(f"immutable candidate fields were changed: {candidate}")
    verdict = pd.to_numeric(candidates.adjudication, errors="coerce")
    bad_verdict = verdict.isna() | ~verdict.isin([0, 1])
    reasons = candidates.reason.astype(str).str.strip()
    bad_reason = reasons.eq("")
    if bad_verdict.any() or bad_reason.any():
        bad = candidates.loc[bad_verdict | bad_reason, "candidate_id"].astype(str).tolist()
        raise ReviewError(
            "every candidate requires adjudication 0/1 and a non-empty reason; "
            f"bad={bad[:8]}"
        )
    output = candidates.copy()
    output["adjudication"] = verdict.astype(int)
    output["reason"] = reasons
    return output


def _records_by_key(sidecar: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = sidecar.get("records")
    if not isinstance(rows, list):
        raise ReviewError("merge sidecar records are missing")
    answer = {(str(row["scene"]), str(row["model"])): row for row in rows}
    if len(answer) != len(rows):
        raise ReviewError("merge sidecar contains duplicate record keys")
    return answer


def adjudicated_event_table(candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse reviewed candidates to unique timed event decisions."""
    columns = (
        "scene", "model", "time_s", "conjuration_flag",
        "accepted_candidate_ids",
    )
    rows: list[dict[str, Any]] = []
    if len(candidates):
        source = candidates.copy()
        source["time_s"] = pd.to_numeric(source.birth_s, errors="raise")
        for (scene, model, time_s), group in source.groupby(
                ["scene", "model", "time_s"], sort=True, dropna=False):
            accepted_ids = group.loc[
                group.adjudication.astype(int).eq(1), "candidate_id"
            ].astype(str).tolist()
            rows.append({
                "scene": str(scene), "model": str(model),
                "time_s": float(time_s),
                "conjuration_flag": int(bool(accepted_ids)),
                "accepted_candidate_ids": json.dumps(
                    accepted_ids, separators=(",", ":")),
            })
    events = pd.DataFrame(rows, columns=columns)
    if (not events.empty and events.duplicated(
            ["scene", "model", "time_s"]).any()):
        raise ReviewError("final event table contains duplicate event keys")
    return events


def finalize(args: argparse.Namespace) -> None:
    manifest, keys = manifest_table(args.manifest)
    candidates = _load_candidates(args.adjudication)
    sidecar_path = args.merge_sidecar or merge_sidecar_path(args.adjudication)
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"cannot read merge sidecar {sidecar_path}: {exc}") from exc
    candidates = _validate_adjudication(candidates, sidecar, sha256_file(args.manifest))
    records = _records_by_key(sidecar)
    if set(records) != keys or int(sidecar.get("manifest_video_keys", -1)) != len(keys):
        raise ReviewError("merge sidecar does not contain the exact manifest video keys")
    if int(sidecar.get("window_rows_expected", -1)) != len(keys) * len(WINDOWS):
        raise ReviewError("merge sidecar window-row contract mismatch")
    for key, metadata in records.items():
        record_path = Path(str(metadata.get("record_path", "")))
        expected_sha = str(metadata.get("record_sha256", ""))
        if (not record_path.is_file() or len(expected_sha) != 64 or
                sha256_file(record_path) != expected_sha):
            raise ReviewError(f"source audit record changed or disappeared: {key}")

    accepted = candidates[candidates.adjudication == 1]
    output_rows = []
    for video in manifest.itertuples(index=False):
        key = (str(video.scene), str(video.model))
        subset = accepted[(accepted.scene == key[0]) & (accepted.model == key[1])]
        for start in WINDOWS:
            ids = subset[subset.source_window_start_s.astype(int) == start].candidate_id.tolist()
            output_rows.append({
                "scene": key[0], "model": key[1], "window_start_s": start,
                "window_end_s": start + WINDOW_SECONDS,
                "conjuration_flag": int(bool(ids)),
                "conjuration_onset_flag": int(bool(ids)),
                "accepted_candidate_ids": json.dumps(ids, separators=(",", ":")),
                "video_sha256": records[key]["video_sha256"],
                "instrument_profile": "panel32-conjuration-v2-human-adjudicated",
            })
    windows = pd.DataFrame(output_rows)
    expected = len(manifest) * len(WINDOWS)
    if len(windows) != expected or windows.duplicated(
            ["scene", "model", "window_start_s"]).any():
        raise ReviewError(f"final window grid is not exact: {len(windows)}/{expected}")

    # Preserve the adjudicated birth time for the cumulative six-axis report.
    # The normalized onset observation uses the first six-second window, while
    # accepted births in later windows enter the persistent conjuration band at
    # their first subsequent endpoint.  Several detector candidates can share
    # an exact birth frame, so emit one unambiguous row per
    # (scene, model, time) and retain all contributing candidate IDs.
    events = adjudicated_event_table(candidates)

    summary_rows = []
    for model in manifest.model.astype(str).drop_duplicates():
        model_windows = windows[windows.model == model]
        model_candidates = candidates[candidates.model == model]
        accepted_model = accepted[accepted.model == model]
        video_count = int((manifest.model.astype(str) == model).sum())
        cumulative: set[str] = set()
        for start in WINDOWS:
            current = model_windows[model_windows.window_start_s == start]
            new_scenes = set(current.loc[current.conjuration_onset_flag == 1, "scene"].astype(str))
            cumulative |= new_scenes
            source_candidates = model_candidates[
                model_candidates.source_window_start_s.astype(int) == start]
            accepted_candidates = accepted_model[
                accepted_model.source_window_start_s.astype(int) == start]
            summary_rows.append({
                "model": model, "window_start_s": start,
                "window_end_s": start + WINDOW_SECONDS, "videos": video_count,
                "candidates_reviewed": len(source_candidates),
                "accepted_candidates": len(accepted_candidates),
                "new_onset_videos": len(new_scenes),
                "onset_rate_pct": 100.0 * len(new_scenes) / video_count,
                "cumulative_flagged_videos": len(cumulative),
                "cumulative_rate_pct": 100.0 * len(cumulative) / video_count,
            })
    model_summary = pd.DataFrame(summary_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_csv(windows, args.output / "conjuration_v2_windows.csv")
    atomic_csv(candidates, args.output / "conjuration_v2_adjudication.csv")
    atomic_csv(events, args.output / "conjuration_v2_events_adjudicated.csv")
    atomic_csv(model_summary, args.output / "conjuration_v2_model_window_summary.csv")
    summary = {
        "schema_version": 1, "status": "pass", "videos": len(manifest),
        "windows_per_video": len(WINDOWS), "window_rows": len(windows),
        "candidates_reviewed": len(candidates),
        "accepted_candidates": int((candidates.adjudication == 1).sum()),
        "accepted_videos": int(accepted[["scene", "model"]].drop_duplicates().shape[0]),
        "adjudicated_event_times": len(events),
        "accepted_event_times": int(events.conjuration_flag.sum()) if len(events) else 0,
        "models": int(manifest.model.nunique()),
    }
    atomic_text(args.output / "conjuration_v2_summary.json",
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--manifest", type=Path, required=True)
    merge_parser.add_argument("--candidates", type=Path, required=True,
                              help="audit root or its records directory")
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.set_defaults(func=merge)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--manifest", type=Path, required=True)
    render_parser.add_argument("--candidates", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.set_defaults(func=render)
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--manifest", type=Path, required=True)
    final_parser.add_argument("--adjudication", type=Path, required=True)
    final_parser.add_argument("--merge-sidecar", type=Path)
    final_parser.add_argument("--output", type=Path, required=True)
    final_parser.set_defaults(func=finalize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
