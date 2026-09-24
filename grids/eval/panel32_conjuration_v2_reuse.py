"""Reuse corrected-conjuration records whose final videos are byte-identical.

This bridges the accepted-context audit into the final v2 audit tree.  A
record is copied only when its (scene, model) exists in the final manifest and
the final video SHA-256 equals the SHA stored by the original audit.  Evidence
is copied into the new tree and every embedded path is rewritten so the final
merge remains self-contained and passes its strict path checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.manifest, keep_default_na=False)
    if table.duplicated(["scene", "model"]).any():
        raise RuntimeError("final video manifest has duplicate keys")
    final = {
        (str(row.scene), str(row.model)): Path(str(row.path)).resolve()
        for row in table.itertuples()
    }
    records = sorted((args.source_audit / "records").glob("*.json"))
    if not records:
        raise RuntimeError("source audit contains no records")
    evidence_root = args.output_audit / "evidence"
    copied, skipped = 0, []
    digest_cache: dict[Path, str] = {}
    for record_path in records:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        key = (str(payload.get("scene")), str(payload.get("model")))
        video = final.get(key)
        if video is None or not video.is_file():
            skipped.append({"key": key, "reason": "not in final manifest"})
            continue
        if video not in digest_cache:
            digest_cache[video] = sha256(video)
        observed = digest_cache[video]
        if observed != payload.get("video_sha256"):
            skipped.append({"key": key, "reason": "final video hash changed"})
            continue
        payload["video_path"] = str(video)
        for window in payload.get("windows", []):
            for candidate in window.get("candidates", []):
                source = Path(str(candidate.get("evidence_path", ""))).resolve()
                if not source.is_file() or source.stat().st_size == 0:
                    raise RuntimeError(f"missing source evidence: {source}")
                destination = evidence_root / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
                elif sha256(destination) != sha256(source):
                    raise RuntimeError(f"conflicting evidence: {destination}")
                candidate["evidence_path"] = str(destination.resolve())
        atomic_json(args.output_audit / "records" / record_path.name, payload)
        copied += 1

    report = {
        "schema_version": 1,
        "source_records": len(records),
        "reused_records": copied,
        "skipped_records": len(skipped),
        "skipped": skipped,
        "source_audit": str(args.source_audit.resolve()),
        "output_audit": str(args.output_audit.resolve()),
        "manifest": str(args.manifest.resolve()),
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
