"""Relocate a validated panel32 canonical source tree without changing media.

The canonical extractor records absolute paths.  This utility is intentionally
small and fail-closed: it rewrites only path fields, verifies every stream and
boundary image against the recorded SHA-256, and emits a new aggregate source
contract suitable for a different machine or stage root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    require(payload.get("status") == "pass", "input provenance is not passing")
    require(payload.get("complete_panel") is True, "input provenance is incomplete")
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 32, "expected 32 source rows")
    manifest_hash = sha256(args.manifest)
    require(
        manifest_hash == payload.get("manifest_sha256"),
        "manifest hash differs from source provenance",
    )

    root = args.canonical_root.resolve()
    manifest = args.manifest.resolve()
    for row in rows:
        context = row["context_id"]
        canonical = row["canonical"]
        stream = root / "streams" / f"source33_{context}.avi"
        boundary = root / "boundaries" / f"source33_{context}_f32.png"
        require(stream.is_file(), f"missing canonical stream: {stream}")
        require(boundary.is_file(), f"missing boundary image: {boundary}")
        require(
            sha256(stream) == canonical["stream_sha256"],
            f"canonical stream hash mismatch: {context}",
        )
        require(
            sha256(boundary) == canonical["boundary_png_sha256"],
            f"boundary image hash mismatch: {context}",
        )
        canonical["stream"] = str(stream)
        canonical["boundary_png"] = str(boundary)
        row["panel_manifest"] = str(manifest)

    payload["manifest"] = str(manifest)
    payload["output_root"] = str(root)
    atomic_json(args.output, payload)
    print(json.dumps({
        "status": "pass",
        "rows": len(rows),
        "manifest_sha256": manifest_hash,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
