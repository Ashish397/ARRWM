#!/usr/bin/env python3
"""Reuse unchanged external-model generations across a locked panel revision.

Only source-identical contexts are eligible.  The tool verifies the complete
old/new per-video input/action contract, allowing differences solely in panel
and source-provenance identity, then writes the new provenance atomically.
Changed contexts are never copied or relabelled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from panel32_model_inputs import (
    ACTIONS,
    NEUTRAL_PROMPT,
    build_plan,
    materialize,
    sidecar_provenance,
)


MODELS = ("lingbot", "dreamx", "matrixgame2", "minwm", "minwm_ode")
LINEAGE_KEYS = {
    "panel_id",
    "panel_manifest",
    "panel_manifest_sha256",
    "panel_source_provenance",
    "panel_source_provenance_sha256",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {row["context_id"]: row for row in payload["rows"]}


def unchanged_contexts(old_path: Path, new_path: Path) -> set[str]:
    old = source_rows(old_path)
    new = source_rows(new_path)
    result = set()
    for context_id in set(old) & set(new):
        a, b = old[context_id], new[context_id]
        if (
            a["source"]["sha256"] == b["source"]["sha256"]
            and a["source_id"] == b["source_id"]
            and a["canonical"]["stream_sha256"]
            == b["canonical"]["stream_sha256"]
            and a["canonical"]["boundary_png_sha256"]
            == b["canonical"]["boundary_png_sha256"]
            and a["boundary"]["original_rgb24_sha256"]
            == b["boundary"]["original_rgb24_sha256"]
            and a["boundary"]["normalized_rgb24_sha256"]
            == b["boundary"]["normalized_rgb24_sha256"]
        ):
            result.add(context_id)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-provenance", required=True, type=Path)
    parser.add_argument("--new-provenance", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--fleet-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    unchanged = unchanged_contexts(args.old_provenance, args.new_provenance)
    if len(unchanged) != 28:
        raise RuntimeError(f"expected exactly 28 source-identical contexts, got {len(unchanged)}")

    migrated = []
    for model in MODELS:
        output = args.fleet_root / model
        for action in ACTIONS:
            plan = build_plan(
                args.manifest,
                args.new_provenance,
                model,
                action,
                args.stage,
                output,
                prompt=NEUTRAL_PROMPT,
            )
            materialize(plan, args.stage)
            for item in plan["items"]:
                context_id = item["context_id"]
                if context_id not in unchanged:
                    continue
                video = Path(item["stable_output"])
                sidecar = Path(str(video) + ".json")
                if not video.is_file() or not sidecar.is_file():
                    raise RuntimeError(f"missing reused external artifact: {video}")
                payload = load_json(sidecar)
                prior = payload.get("panel32")
                if not isinstance(prior, dict):
                    raise RuntimeError(f"missing prior panel32 block: {sidecar}")
                expected = sidecar_provenance(plan, item)
                prior_stable = {k: v for k, v in prior.items() if k not in LINEAGE_KEYS}
                expected_stable = {k: v for k, v in expected.items() if k not in LINEAGE_KEYS}
                if prior_stable != expected_stable:
                    raise RuntimeError(
                        f"non-lineage contract changed for {model}/{context_id}/{action}"
                    )
                payload["panel32"] = expected
                if "seed_image" in payload:
                    payload["seed_image"] = item["input_path"]
                if "seed_video" in payload:
                    payload["seed_video"] = item["input_path"]
                atomic_json(sidecar, payload)
                migrated.append({
                    "model": model,
                    "context_id": context_id,
                    "action": action,
                    "video": str(video),
                })

    expected_count = len(MODELS) * len(ACTIONS) * len(unchanged)
    if len(migrated) != expected_count:
        raise RuntimeError(f"migrated {len(migrated)}, expected {expected_count}")
    report = {
        "status": "pass",
        "unchanged_contexts": sorted(unchanged),
        "models": list(MODELS),
        "actions": list(ACTIONS),
        "migrated_sidecars": len(migrated),
        "rows": migrated,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
