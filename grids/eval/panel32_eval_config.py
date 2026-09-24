"""Build the single resolver configuration for the complete panel32 rerun."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from .panel32_manifest import load_panel_manifest, sha256_file
except ImportError:
    from panel32_manifest import load_panel_manifest, sha256_file  # type: ignore


SCHEMA_VERSION = 1
ARM_TAGS = {
    "ours_recovery_base": "recoverybase",
    "ours_no_carn": "nocarn",
    "ours_no_commit": "nocommit",
    "ours_no_aux": "noaux",
    "ours_no_gan": "nogan",
    "ours_stat_mean_only": "meanenergy",
    "ours_stat_nonmean_only": "vartv",
    "ours_kl4rung": "kl4rung",
    "ours_mse4rung": "mse4rung",
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(manifest_path: Path, provenance_path: Path, fleet_root: Path) -> dict[str, Any]:
    panel = load_panel_manifest(manifest_path, verify_sources=False)
    provenance = json.loads(provenance_path.read_text())
    if (
        provenance.get("status") != "pass"
        or provenance.get("complete_panel") is not True
        or provenance.get("manifest_sha256") != panel.sha256
    ):
        raise ValueError("source provenance is not a complete pass for this manifest")
    rows = provenance.get("rows", [])
    by_id = {row.get("context_id"): row for row in rows if isinstance(row, dict)}
    ids = [context.context_id for context in panel.contexts]
    if len(rows) != 32 or set(by_id) != set(ids):
        raise ValueError("source provenance does not contain the exact 32 contexts")
    clips = {
        context_id: str(Path(by_id[context_id]["canonical"]["stream"]).resolve())
        for context_id in ids
    }
    for context_id, clip in clips.items():
        if not Path(clip).is_file():
            raise ValueError(f"missing canonical source clip for {context_id}: {clip}")

    root = fleet_root.resolve()
    models: dict[str, dict[str, Any]] = {
        "lingbot": dict(family="lingbot", context_frames=1,
            source_frames_inclusive=[32, 32],
            path_template=str(root / "lingbot/lingbot_{context_id}_{action}.mp4")),
        "dreamx": dict(family="dreamx", context_frames=1,
            source_frames_inclusive=[32, 32],
            path_template=str(root / "dreamx/dreamx_{context_id}_{action}.mp4")),
        "matrixgame2": dict(family="matrixgame2", context_frames=1,
            source_frames_inclusive=[32, 32], noop_disk_action="NOOP",
            path_template=str(root / "matrixgame2/matrixgame_{context_id}_{action}.mp4")),
        "minwm": dict(family="minwm", context_frames=29,
            source_frames_inclusive=[4, 32], noop_disk_action="NOOP",
            path_template=str(root / "minwm/minwm_aligned32_{context_id}_{action}.mp4")),
        "minwm_ode": dict(family="minwm", context_frames=13,
            source_frames_inclusive=[20, 32], noop_disk_action="NOOP",
            path_template=str(root / "minwm_ode/minwm_ode_{context_id}_{action}.mp4")),
        "yume5b": dict(family="yume5b", context_frames=1,
            source_frames_inclusive=[32, 32],
            path_template=str(root / "yume5b/yume5b_{context_id}_{action}.mp4")),
    }
    for model, tag in ARM_TAGS.items():
        models[model] = dict(
            family="ours", context_frames=33,
            source_frames_inclusive=[0, 32],
            # Generation directories use the compact, underscore-free tag so
            # every arm follows the same resumable filename contract.
            path_template=str(root / f"ours_{tag}" / f"{tag}_{{context_id}}_{{action}}.mp4"),
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": panel.panel_id,
        "panel_manifest": str(panel.path),
        "panel_manifest_sha256": panel.sha256,
        "source_provenance": str(provenance_path.resolve()),
        "source_provenance_sha256": sha256_file(provenance_path),
        "fleet_root": str(root),
        "context_ids": ids,
        "source_clips": clips,
        "models": models,
        "family_seats": {
            "ours": "ours_no_gan",
            "lingbot": "lingbot",
            "dreamx": "dreamx",
            "matrixgame2": "matrixgame2",
            "minwm": "minwm",
            "yume5b": "yume5b",
        },
        "main_models": [
            "ours_no_gan", "lingbot", "dreamx", "matrixgame2",
            "minwm", "yume5b",
        ],
        "ode_models": ["ours_kl4rung", "ours_mse4rung", "minwm_ode"],
        "ablation_models": [
            "ours_no_gan", "ours_no_carn", "ours_no_commit",
            "ours_stat_mean_only",
            "ours_stat_nonmean_only",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-provenance", required=True, type=Path)
    parser.add_argument("--fleet-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build(args.manifest, args.source_provenance, args.fleet_root)
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({
        "output": str(args.output.resolve()), "contexts": len(payload["context_ids"]),
        "models": len(payload["models"]), "families": len(payload["family_seats"]),
    }, indent=2))


if __name__ == "__main__":
    main()
