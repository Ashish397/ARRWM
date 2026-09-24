"""Schema and fail-closed validation for the mixed 32-context panel.

This module intentionally has no dependency on the FrodoBots evaluation
manifests.  A panel contains exactly eight contexts from each of FrodoBots,
Ego4D, Sekai, and SpatialVID.  Context identifiers are dataset-prefixed and
safe to use in filenames.

The canonical source stream is the *input* to the later Wan seed encoder.  Its
33 frames satisfy Wan's temporal packing relation for nine latent frames::

    pixel_frames = 1 + 4 * (latent_frames - 1)

The source boundary (an original frame or timestamp) is kept distinct from
canonical frame 32.  This is important for variable-frame-rate source media.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = 1
DATASET_COUNTS = {
    "FrodoBots": 8,
    "Ego4D": 8,
    "Sekai": 8,
    "SpatialVID": 8,
}
CANONICAL_FRAME_COUNT = 33
CANONICAL_FPS = 20.0
CANONICAL_WIDTH = 832
CANONICAL_HEIGHT = 480
CANONICAL_BOUNDARY_FRAME_INDEX = 32
WAN_LATENT_FRAMES = 9
ACTIONS = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
DATASET_PREFIX = {
    "FrodoBots": "frodobots-",
    "Ego4D": "ego4d-",
    "Sekai": "sekai-",
    "SpatialVID": "spatialvid-",
}
CONTEXT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESIZE_MODES = {"center_crop", "stretch"}


PANEL32_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ARRWM mixed 32-context evaluation panel",
    "type": "object",
    "required": [
        "schema_version", "panel_id", "selection_protocol", "created_utc",
        "dataset_counts", "actions", "canonical_source", "contexts",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "panel_id": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "selection_protocol": {"type": "string"},
        "created_utc": {"type": "string"},
        "dataset_counts": {"const": DATASET_COUNTS},
        "actions": {"const": list(ACTIONS)},
        "canonical_source": {
            "type": "object",
            "required": [
                "frame_count", "fps", "width", "height", "resize_mode",
                "boundary_frame_index", "wan_latent_frames",
            ],
            "additionalProperties": False,
            "properties": {
                "frame_count": {"const": CANONICAL_FRAME_COUNT},
                "fps": {"const": CANONICAL_FPS},
                "width": {"const": CANONICAL_WIDTH},
                "height": {"const": CANONICAL_HEIGHT},
                "resize_mode": {"enum": sorted(RESIZE_MODES)},
                "boundary_frame_index": {"const": CANONICAL_BOUNDARY_FRAME_INDEX},
                "wan_latent_frames": {"const": WAN_LATENT_FRAMES},
            },
        },
        "contexts": {
            "type": "array",
            "minItems": 32,
            "maxItems": 32,
            "items": {
                "type": "object",
                "required": [
                    "context_id", "dataset", "source_id", "source_video",
                    "source_sha256", "source_uri", "license", "boundary",
                    "outdoor_verified", "outdoor_verification_note",
                ],
                "additionalProperties": False,
                "properties": {
                    "context_id": {"type": "string", "pattern": CONTEXT_ID_RE.pattern},
                    "dataset": {"enum": list(DATASET_COUNTS)},
                    "source_id": {"type": "string", "minLength": 1},
                    "source_video": {"type": "string", "minLength": 1},
                    "source_sha256": {"type": "string", "pattern": SHA256_RE.pattern},
                    "source_uri": {"type": "string"},
                    "license": {"type": "string"},
                    "selection_note": {"type": "string"},
                    "outdoor_verified": {"const": True},
                    "outdoor_verification_note": {"type": "string", "minLength": 1},
                    "boundary": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 1,
                        "additionalProperties": False,
                        "properties": {
                            "frame_index": {"type": "integer", "minimum": 0},
                            "timestamp_s": {"type": "number", "minimum": 0},
                        },
                    },
                },
            },
        },
    },
    "additionalProperties": False,
}


class ManifestError(ValueError):
    """The panel manifest is incomplete, ambiguous, or internally invalid."""


@dataclass(frozen=True)
class CanonicalSourceSpec:
    frame_count: int
    fps: float
    width: int
    height: int
    resize_mode: str
    boundary_frame_index: int
    wan_latent_frames: int


@dataclass(frozen=True)
class ContextSpec:
    context_id: str
    dataset: str
    source_id: str
    declared_source_video: str
    source_video: Path
    source_sha256: str
    boundary_kind: str
    boundary_value: float | int
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class PanelManifest:
    path: Path
    sha256: str
    panel_id: str
    canonical: CanonicalSourceSpec
    contexts: tuple[ContextSpec, ...]
    raw: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _exact_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    _require(not unknown, f"{where}: unknown keys {sorted(unknown)}")


def _canonical_spec(payload: Mapping[str, Any]) -> CanonicalSourceSpec:
    _require(isinstance(payload, Mapping), "canonical_source must be an object")
    required = {
        "frame_count", "fps", "width", "height", "resize_mode",
        "boundary_frame_index", "wan_latent_frames",
    }
    _require(required <= set(payload),
             f"canonical_source: missing keys {sorted(required - set(payload))}")
    _exact_keys(payload, required, "canonical_source")
    for key in ("frame_count", "width", "height", "boundary_frame_index",
                "wan_latent_frames"):
        _require(isinstance(payload[key], int) and not isinstance(payload[key], bool),
                 f"canonical_source.{key} must be an integer")
    _require(isinstance(payload["fps"], (int, float))
             and not isinstance(payload["fps"], bool),
             "canonical_source.fps must be numeric")
    _require(isinstance(payload["resize_mode"], str),
             "canonical_source.resize_mode must be a string")
    spec = CanonicalSourceSpec(
        frame_count=int(payload["frame_count"]),
        fps=float(payload["fps"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        resize_mode=str(payload["resize_mode"]),
        boundary_frame_index=int(payload["boundary_frame_index"]),
        wan_latent_frames=int(payload["wan_latent_frames"]),
    )
    _require(spec.frame_count == CANONICAL_FRAME_COUNT,
             f"canonical_source.frame_count must be {CANONICAL_FRAME_COUNT}")
    _require(spec.fps == CANONICAL_FPS,
             f"canonical_source.fps must be {CANONICAL_FPS:g}")
    _require(spec.width == CANONICAL_WIDTH,
             f"canonical_source.width must be {CANONICAL_WIDTH}")
    _require(spec.height == CANONICAL_HEIGHT,
             f"canonical_source.height must be {CANONICAL_HEIGHT}")
    _require(spec.boundary_frame_index == CANONICAL_BOUNDARY_FRAME_INDEX,
             "canonical_source.boundary_frame_index must be "
             f"{CANONICAL_BOUNDARY_FRAME_INDEX}")
    _require(spec.wan_latent_frames == WAN_LATENT_FRAMES,
             f"canonical_source.wan_latent_frames must be {WAN_LATENT_FRAMES}")
    _require(1 + 4 * (spec.wan_latent_frames - 1) == spec.frame_count,
             "canonical_source violates Wan temporal packing")
    _require(math.isfinite(spec.fps) and spec.fps > 0,
             "canonical_source.fps must be positive and finite")
    _require(spec.width > 0 and spec.height > 0,
             "canonical_source dimensions must be positive")
    _require(spec.width % 2 == 0 and spec.height % 2 == 0,
             "canonical_source dimensions must be even")
    _require(spec.resize_mode in RESIZE_MODES,
             f"canonical_source.resize_mode must be one of {sorted(RESIZE_MODES)}")
    return spec


def _context_spec(row: Mapping[str, Any], manifest_dir: Path) -> ContextSpec:
    _require(isinstance(row, Mapping), "every context row must be an object")
    required = {"context_id", "dataset", "source_id", "source_video",
                "source_sha256", "source_uri", "license", "boundary",
                "outdoor_verified", "outdoor_verification_note"}
    optional = {"selection_note"}
    _require(required <= set(row),
             f"context: missing keys {sorted(required - set(row))}")
    _exact_keys(row, required | optional, f"context {row.get('context_id', '?')}")
    for key in ("context_id", "dataset", "source_id", "source_video",
                "source_sha256"):
        _require(isinstance(row[key], str), f"context.{key} must be a string")
    for key in {"source_uri", "license", "outdoor_verification_note"} | (optional & set(row)):
        _require(isinstance(row[key], str),
                 f"{row['context_id']}.{key} must be a string")
    context_id = row["context_id"]
    dataset = row["dataset"]
    _require(dataset in DATASET_COUNTS, f"{context_id}: unknown dataset {dataset!r}")
    _require(CONTEXT_ID_RE.fullmatch(context_id) is not None,
             f"{context_id!r}: context_id is not filename-safe lowercase ASCII")
    _require(context_id.startswith(DATASET_PREFIX[dataset]),
             f"{context_id}: must start with {DATASET_PREFIX[dataset]!r}")
    source_id = row["source_id"].strip()
    _require(bool(source_id), f"{context_id}: source_id is empty")
    _require(bool(row["source_uri"].strip()), f"{context_id}: source_uri is empty")
    _require(bool(row["license"].strip()), f"{context_id}: license is empty")
    _require(row["outdoor_verified"] is True,
             f"{context_id}: outdoor_verified must be true")
    _require(bool(row["outdoor_verification_note"].strip()),
             f"{context_id}: outdoor_verification_note is empty")
    declared = row["source_video"]
    _require(bool(declared), f"{context_id}: source_video is empty")
    source = Path(declared).expanduser()
    if not source.is_absolute():
        source = manifest_dir / source
    source = source.resolve()
    source_hash = row["source_sha256"]
    _require(SHA256_RE.fullmatch(source_hash) is not None,
             f"{context_id}: source_sha256 must be 64 lowercase hex characters")

    boundary = row["boundary"]
    _require(isinstance(boundary, Mapping), f"{context_id}: boundary must be an object")
    _exact_keys(boundary, {"frame_index", "timestamp_s"}, f"{context_id}.boundary")
    _require(len(boundary) == 1,
             f"{context_id}: boundary must contain exactly one of frame_index/timestamp_s")
    if "frame_index" in boundary:
        value = boundary["frame_index"]
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 f"{context_id}: boundary.frame_index must be a non-negative integer")
        boundary_kind, boundary_value = "frame_index", int(value)
    else:
        value = boundary["timestamp_s"]
        _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                 f"{context_id}: boundary.timestamp_s must be numeric")
        value = float(value)
        _require(math.isfinite(value) and value >= 0,
                 f"{context_id}: boundary.timestamp_s must be non-negative and finite")
        boundary_kind, boundary_value = "timestamp_s", value
    return ContextSpec(
        context_id=context_id,
        dataset=dataset,
        source_id=source_id,
        declared_source_video=declared,
        source_video=source,
        source_sha256=source_hash,
        boundary_kind=boundary_kind,
        boundary_value=boundary_value,
        raw=row,
    )


def load_panel_manifest(path: str | Path, *, verify_sources: bool = False) -> PanelManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read panel manifest {manifest_path}: {exc}") from exc
    _require(isinstance(payload, Mapping), "panel manifest root must be an object")
    required = {"schema_version", "panel_id", "selection_protocol", "created_utc",
                "dataset_counts", "actions", "canonical_source", "contexts"}
    optional = {"description"}
    _require(required <= set(payload),
             f"panel manifest: missing keys {sorted(required - set(payload))}")
    _exact_keys(payload, required | optional, "panel manifest")
    _require(payload["schema_version"] == SCHEMA_VERSION,
             f"schema_version must be {SCHEMA_VERSION}")
    _require(isinstance(payload["panel_id"], str), "panel_id must be a string")
    panel_id = payload["panel_id"].strip()
    _require(bool(panel_id), "panel_id is empty")
    for key in ("selection_protocol", "created_utc"):
        _require(isinstance(payload[key], str) and bool(payload[key].strip()),
                 f"{key} must be a non-empty string")
    _require(payload["dataset_counts"] == DATASET_COUNTS,
             f"dataset_counts must be exactly {DATASET_COUNTS}")
    _require(payload["actions"] == list(ACTIONS),
             f"actions must be exactly {list(ACTIONS)}")
    canonical = _canonical_spec(payload["canonical_source"])
    rows = payload["contexts"]
    _require(isinstance(rows, list) and len(rows) == 32,
             "contexts must contain exactly 32 rows")
    contexts = tuple(_context_spec(row, manifest_path.parent) for row in rows)
    ids = [row.context_id for row in contexts]
    _require(len(set(ids)) == len(ids), "context_id values must be globally unique")
    source_keys = [(row.dataset, row.source_id) for row in contexts]
    _require(len(set(source_keys)) == len(source_keys),
             "(dataset, source_id) values must be unique")
    counts = Counter(row.dataset for row in contexts)
    _require(dict(counts) == DATASET_COUNTS,
             f"context dataset counts are {dict(counts)}, expected {DATASET_COUNTS}")
    if verify_sources:
        for row in contexts:
            _require(row.source_video.is_file(),
                     f"{row.context_id}: source video missing: {row.source_video}")
            observed = sha256_file(row.source_video)
            _require(observed == row.source_sha256,
                     f"{row.context_id}: source SHA mismatch: {observed} != {row.source_sha256}")
    return PanelManifest(
        path=manifest_path,
        sha256=sha256_file(manifest_path),
        panel_id=panel_id,
        canonical=canonical,
        contexts=contexts,
        raw=payload,
    )


def validation_summary(panel: PanelManifest) -> dict[str, Any]:
    return {
        "status": "pass",
        "schema_version": SCHEMA_VERSION,
        "panel_id": panel.panel_id,
        "manifest": str(panel.path),
        "manifest_sha256": panel.sha256,
        "contexts": len(panel.contexts),
        "dataset_counts": dict(Counter(row.dataset for row in panel.contexts)),
        "actions": list(ACTIONS),
        "canonical_source": {
            "frame_count": panel.canonical.frame_count,
            "fps": panel.canonical.fps,
            "width": panel.canonical.width,
            "height": panel.canonical.height,
            "resize_mode": panel.canonical.resize_mode,
            "boundary_frame_index": panel.canonical.boundary_frame_index,
            "wan_latent_frames": panel.canonical.wan_latent_frames,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", action="store_true", help="print the JSON schema helper")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(PANEL32_JSON_SCHEMA, indent=2, sort_keys=True))
        return
    if args.manifest is None:
        parser.error("--manifest is required unless --schema is used")
    panel = load_panel_manifest(args.manifest, verify_sources=args.verify_sources)
    print(json.dumps(validation_summary(panel), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
