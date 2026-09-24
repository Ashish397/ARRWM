"""Audit Matrix-Game media and compare collapse behaviour across devices.

Black generated states are retained as outcomes.  This audit distinguishes
them from missing/short media, a black conditioning image, an action/seed
mismatch, or a device-specific result.  It never repairs or excludes a video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ACTIONS = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
SAMPLE_SECONDS = tuple(range(0, 31))


class AuditError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _decode(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise AuditError(f"cannot decode frame {index}: {path}")
    return image


def _sample_video(path: Path, frames: int, fps: float) -> list[dict[str, Any]]:
    """Decode once and retain only the requested per-second sample frames."""
    requested = {
        min(frames - 1, int(round(second * fps))): second
        for second in SAMPLE_SECONDS
    }
    samples_by_second: dict[int, dict[str, Any]] = {}
    capture = cv2.VideoCapture(str(path))
    try:
        for index in range(max(requested) + 1):
            ok, image = capture.read()
            if not ok or image is None:
                raise AuditError(f"cannot decode frame {index}: {path}")
            second = requested.get(index)
            if second is not None:
                samples_by_second[second] = {
                    "second": second,
                    "frame_index": index,
                    **_frame_stats(image),
                }
    finally:
        capture.release()
    missing = [second for second in SAMPLE_SECONDS if second not in samples_by_second]
    if missing:
        raise AuditError(f"missing requested samples {missing}: {path}")
    return [samples_by_second[second] for second in SAMPLE_SECONDS]


def _frame_stats(image: np.ndarray) -> dict[str, float | bool]:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(grey.mean())
    std = float(grey.std())
    return {
        "grey_mean": mean,
        "grey_std": std,
        "black_screen": bool(mean <= 3.0 and std <= 3.0),
    }


def _sidecar_value(payload: dict[str, Any], name: str) -> Any:
    panel = payload.get("panel32")
    if isinstance(panel, dict) and name in panel:
        return panel[name]
    aliases = {
        "context_id": ("window",),
        "action": ("direction",),
        "sampling_seed": ("seed",),
    }
    if name in payload:
        return payload[name]
    for alias in aliases.get(name, ()):
        if alias in payload:
            return payload[alias]
    return None


def audit_video(path: Path, context_id: str, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing video: {path}")
    sidecar_path = Path(str(path) + ".json")
    if not sidecar_path.is_file():
        raise AuditError(f"missing sidecar: {sidecar_path}")
    payload = json.loads(sidecar_path.read_text())
    observed_context = _sidecar_value(payload, "context_id")
    observed_action = _sidecar_value(payload, "action")
    if observed_context != context_id:
        raise AuditError(
            f"{path}: context {observed_context!r}, expected {context_id!r}"
        )
    if observed_action not in (action, "NOOP" if action == "N" else action):
        raise AuditError(
            f"{path}: action {observed_action!r}, expected {action!r}"
        )
    seed = _sidecar_value(payload, "sampling_seed")
    if str(seed) != "0":
        raise AuditError(f"{path}: sampling seed {seed!r}, expected 0")
    if int(payload.get("latent_frames", -1)) != 189:
        raise AuditError(f"{path}: expected 189 latent frames")

    capture = cv2.VideoCapture(str(path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frames != 753 or not math.isclose(fps, 25.0, abs_tol=0.05):
        raise AuditError(f"{path}: unexpected media geometry {(frames, fps)}")
    if (width, height) != (640, 352):
        raise AuditError(f"{path}: unexpected raster {(width, height)}")

    samples = _sample_video(path, frames, fps)
    if samples[0]["black_screen"]:
        raise AuditError(f"{path}: conditioning frame is black")
    black_seconds = [
        int(row["second"]) for row in samples[1:] if row["black_screen"]
    ]
    return {
        "context_id": context_id,
        "action": action,
        "path": str(path.resolve()),
        "sidecar": str(sidecar_path.resolve()),
        "sampling_seed": seed,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "samples": samples,
        "black_seconds": black_seconds,
        "first_black_second": black_seconds[0] if black_seconds else None,
    }


def audit_fleet(root: Path, contexts: list[str]) -> dict[str, Any]:
    rows = []
    errors = []
    for context_id in contexts:
        for action in ACTIONS:
            disk_action = "NOOP" if action == "N" else action
            path = root / f"matrixgame_{context_id}_{disk_action}.mp4"
            try:
                rows.append(audit_video(path, context_id, action))
            except Exception as exc:
                errors.append({
                    "context_id": context_id,
                    "action": action,
                    "path": str(path),
                    "error": repr(exc),
                })
    expected = len(contexts) * len(ACTIONS)
    return {
        "status": "pass" if not errors and len(rows) == expected else "fail",
        "root": str(root.resolve()),
        "contexts": contexts,
        "expected_videos": expected,
        "audited_videos": len(rows),
        "videos_black_by_second": {
            str(second): sum(
                bool(row["samples"][second]["black_screen"]) for row in rows
            )
            for second in SAMPLE_SECONDS[1:]
        },
        "videos_ever_black_by_second": {
            str(second): sum(
                row["first_black_second"] is not None
                and int(row["first_black_second"]) <= second
                for row in rows
            )
            for second in SAMPLE_SECONDS[1:]
        },
        "rows": rows,
        "errors": errors,
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {
        (row["context_id"], row["action"]): row for row in left["rows"]
    }
    right_rows = {
        (row["context_id"], row["action"]): row for row in right["rows"]
    }
    expected = {
        (context, action) for context in left["contexts"] for action in ACTIONS
    }
    errors = []
    if set(left_rows) != expected or set(right_rows) != expected:
        errors.append("one or both fleets do not contain the exact expected key set")
    mismatches = []
    compared = 0
    agreements = 0
    for key in sorted(set(left_rows) & set(right_rows)):
        lrow, rrow = left_rows[key], right_rows[key]
        for second in SAMPLE_SECONDS:
            lblack = bool(lrow["samples"][second]["black_screen"])
            rblack = bool(rrow["samples"][second]["black_screen"])
            compared += 1
            agreements += int(lblack == rblack)
            if lblack != rblack:
                mismatches.append({
                    "context_id": key[0],
                    "action": key[1],
                    "second": second,
                    "left_black": lblack,
                    "right_black": rblack,
                    "left_mean": lrow["samples"][second]["grey_mean"],
                    "right_mean": rrow["samples"][second]["grey_mean"],
                })
    # This deliberately strict label only establishes black-state parity.  It
    # does not authorize mixing devices; metric-decision parity is a separate
    # required gate after the H100 evaluation completes.
    return {
        "status": (
            "pass" if left["status"] == right["status"] == "pass"
            and not errors and not mismatches else "fail"
        ),
        "scope": "media validity and per-second black-state parity only",
        "compared_sample_states": compared,
        "matching_sample_states": agreements,
        "agreement_fraction": agreements / compared if compared else 0.0,
        "mismatches": mismatches,
        "errors": errors,
    }


def main() -> None:
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", required=True, type=Path)
    parser.add_argument("--contexts", required=True,
                        help="comma-separated context IDs")
    parser.add_argument("--reference-fleet", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    contexts = [value.strip() for value in args.contexts.split(",") if value.strip()]
    if not contexts or len(contexts) != len(set(contexts)):
        raise SystemExit("--contexts must contain distinct context IDs")
    primary = audit_fleet(args.fleet.resolve(), contexts)
    payload: dict[str, Any] = {"primary": primary}
    if args.reference_fleet:
        reference = audit_fleet(args.reference_fleet.resolve(), contexts)
        payload["reference"] = reference
        payload["comparison"] = compare(primary, reference)
        payload["status"] = payload["comparison"]["status"]
    else:
        payload["status"] = primary["status"]
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "primary": {key: value for key, value in primary.items()
                    if key not in ("rows", "errors")},
        "primary_errors": primary["errors"],
        "comparison": payload.get("comparison"),
    }, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
