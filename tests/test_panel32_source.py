from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from grids.eval.panel32_manifest import ManifestError, load_panel_manifest, sha256_file
from grids.eval.panel32_source import decode_canonical, extract_panel, probe_source


DATASETS = ("FrodoBots", "Ego4D", "Sekai", "SpatialVID")
PREFIX = {
    "FrodoBots": "frodobots",
    "Ego4D": "ego4d",
    "Sekai": "sekai",
    "SpatialVID": "spatialvid",
}


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)


def make_source(path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=96x64:rate=30", "-frames:v", "90",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", str(path),
    ], check=True)


def make_manifest(path: Path, source: Path) -> dict:
    digest = sha256_file(source)
    contexts = []
    for dataset in DATASETS:
        for index in range(8):
            boundary = ({"frame_index": 60} if index % 2 == 0
                        else {"timestamp_s": 2.01})
            contexts.append({
                "context_id": f"{PREFIX[dataset]}-c{index:02d}",
                "dataset": dataset,
                "source_id": f"{PREFIX[dataset]}-source-{index:02d}",
                "source_video": source.name,
                "source_sha256": digest,
                "source_uri": f"https://example.invalid/{dataset}/{index}",
                "license": "synthetic test fixture",
                "outdoor_verified": True,
                "outdoor_verification_note": "synthetic test pattern marked outdoor for schema testing",
                "boundary": boundary,
                "selection_note": "synthetic test context",
            })
    payload = {
        "schema_version": 1,
        "panel_id": "synthetic-panel32-v1",
        "selection_protocol": "fixed before model generation",
        "created_utc": "2026-09-23T00:00:00Z",
        "dataset_counts": {dataset: 8 for dataset in DATASETS},
        "actions": ["F", "FR", "R", "BR", "B", "BL", "L", "FL", "N"],
        "canonical_source": {
            "frame_count": 33,
            "fps": 20,
            "width": 832,
            "height": 480,
            "resize_mode": "center_crop",
            "boundary_frame_index": 32,
            "wan_latent_frames": 9,
        },
        "contexts": contexts,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def test_manifest_enforces_balanced_prefixed_panel(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source)
    manifest = tmp_path / "panel.json"
    payload = make_manifest(manifest, source)
    panel = load_panel_manifest(manifest, verify_sources=True)
    assert len(panel.contexts) == 32
    assert panel.canonical.frame_count == 33
    assert panel.canonical.wan_latent_frames == 9

    payload["canonical_source"]["fps"] = 16
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="fps must be 20"):
        load_panel_manifest(manifest)

    payload["canonical_source"]["fps"] = 20
    payload["contexts"][0]["context_id"] = "u31"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="must start"):
        load_panel_manifest(manifest)

    payload["contexts"][0]["context_id"] = "frodobots-c00"
    payload["contexts"][0]["outdoor_verified"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="outdoor_verified must be true"):
        load_panel_manifest(manifest)


def test_extracts_exact_boundary_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source)
    manifest = tmp_path / "panel.json"
    make_manifest(manifest, source)
    selected = ["frodobots-c00", "ego4d-c01"]

    first = extract_panel(manifest, tmp_path / "out-a", context_ids=selected)
    retained = extract_panel(manifest, tmp_path / "out-a", context_ids=selected)
    second = extract_panel(manifest, tmp_path / "out-b", context_ids=selected)
    assert first["complete_panel"] is False
    assert first["extracted_contexts"] == 2
    assert [row["canonical"]["stream_sha256"] for row in retained["rows"]] == [
        row["canonical"]["stream_sha256"] for row in first["rows"]
    ]
    for row_a, row_b in zip(first["rows"], second["rows"]):
        assert row_a["boundary"]["source_frame_index"] == 60
        assert row_a["boundary"]["source_timestamp_s"] == pytest.approx(2.0)
        assert row_a["sampling"]["source_frame_indices"][-1] == 60
        assert len(row_a["sampling"]["source_frame_indices"]) == 33
        assert row_a["canonical"]["frame_count"] == 33
        assert row_a["wan_interface"]["latent_frames"] == 9
        # FFV1/AVI output and decoded RGB are byte-deterministic.
        assert row_a["canonical"]["stream_sha256"] == row_b["canonical"]["stream_sha256"]
        assert (row_a["canonical"]["decoded_rgb24_sha256"]
                == row_b["canonical"]["decoded_rgb24_sha256"])
        stream = Path(row_a["canonical"]["stream"])
        decoded = decode_canonical(stream, 832, 480)
        assert len(decoded) == 33 * 832 * 480 * 3
        final = decoded[-832 * 480 * 3:]
        assert hashlib.sha256(final).hexdigest() == row_a["boundary"]["normalized_rgb24_sha256"]
        assert probe_source(stream)["decoded_frames"] == 33


def test_refuses_insufficient_causal_history(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source)
    manifest = tmp_path / "panel.json"
    payload = make_manifest(manifest, source)
    payload["contexts"][0]["boundary"] = {"frame_index": 20}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="insufficient causal history"):
        extract_panel(
            manifest, tmp_path / "out", context_ids=["frodobots-c00"])


def test_refuses_timestamp_beyond_source(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source)
    manifest = tmp_path / "panel.json"
    payload = make_manifest(manifest, source)
    payload["contexts"][0]["boundary"] = {"timestamp_s": 99.0}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="beyond source end"):
        extract_panel(
            manifest, tmp_path / "out", context_ids=["frodobots-c00"])
