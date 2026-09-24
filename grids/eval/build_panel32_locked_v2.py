#!/usr/bin/env python3
"""Build the final balanced outdoor panel after author-approved replacements.

The v1 manifest remains immutable.  This builder replaces the seven rejected
Ego4D review clips, the three rejected Sekai clips, and all eight symbol-bearing
SpatialVID renders while retaining the eight FrodoBots contexts and the one
approved v1 Ego4D context.  No model output or metric is consulted here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from build_panel32_working_v2 import SEKAI


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "grids/eval/panel32_locked_v1.json"
SPATIAL_PATCH = ROOT / "grids/eval/spatialvid_clean_source_patch_v2.json"
OUTPUT = ROOT / "grids/eval/panel32_locked_v2.json"


EGO4D = {
    "ego4d-bristol-01": {
        "context_id": "ego4d-8ed9e028",
        "q_uid": "8ed9e028-c417-4eda-9e31-f61f60e50ae9",
        "google_drive_id": "1JyWxqS5aYIWlLKQmaD1yk2b9mS492Cal",
        "ego4d_video_uid": "9b34e344-494d-4664-8180-a8ff2a692ab2",
        "ego4d_starting_sec": 240,
        "ego4d_ending_sec": 420,
        "review_file": "../../iclr/replacement_candidates/ego4d_round2/045_8ed9e028-c417-4eda-9e31-f61f60e50ae9.mp4",
        "sha256": "6853e1d1268399141900f3f94e74a497dfec1318667ec66bc6d9a5a0bd73d14c",
    },
    "ego4d-cmu-05": {
        "context_id": "ego4d-fe3d59ac",
        "q_uid": "fe3d59ac-9686-44d0-8b67-ddc08d36a615",
        "google_drive_id": "1qhGa3R6BStszZRzbf7mFAn3KUgtGfgp6",
        "ego4d_video_uid": "b849a2d6-3eff-4c64-aded-2d706ed1091e",
        "ego4d_starting_sec": 0,
        "ego4d_ending_sec": 180,
        "review_file": "../../iclr/replacement_candidates/ego4d_round2/008_fe3d59ac-9686-44d0-8b67-ddc08d36a615.mp4",
        "sha256": "1aed56164d62387dbaf80150bb8d957a9a2e832167c64485961e4b6a76cff300",
    },
    "ego4d-iiith-02": {
        "context_id": "ego4d-97fd209a",
        "q_uid": "97fd209a-1de4-4c85-9815-cb6295b45fdb",
        "google_drive_id": "1idIoNGlfRAyfxNKc6zTs-R3tZUEq2y5h",
        "ego4d_video_uid": "863011c8-ebf7-4900-87d1-3c7930f95dab",
        "ego4d_starting_sec": 3270,
        "ego4d_ending_sec": 3450,
        "review_file": "../../iclr/replacement_candidates/ego4d_round3/005_97fd209a-1de4-4c85-9815-cb6295b45fdb.mp4",
        "sha256": "7bcc2123eef8931da93f8a4095881ed538692e3cf265e3b5e911002e53d6858b",
    },
    "ego4d-kaust-02": {
        "context_id": "ego4d-0bd9a251",
        "q_uid": "0bd9a251-cc2c-41ea-9a6c-31cda3519371",
        "google_drive_id": "1VLSqjMBGOhS-XfdFO0P1jCgLn__z1zne",
        "ego4d_video_uid": "5ead8e7c-2600-44b1-afd6-b82cb5f7b9ce",
        "ego4d_starting_sec": 3210,
        "ego4d_ending_sec": 3390,
        "review_file": "../../iclr/replacement_candidates/ego4d_round3/010_0bd9a251-cc2c-41ea-9a6c-31cda3519371.mp4",
        "sha256": "50bed95e69c1562c3c416a886dda581314ea4322bf7759569317591d30ea0b7f",
    },
    "ego4d-kaust-03": {
        "context_id": "ego4d-de54e5c6",
        "q_uid": "de54e5c6-711b-46a2-abeb-be8671870883",
        "google_drive_id": "13gdInpW4616nXoKjxiFrBFo3t47QAgyW",
        "ego4d_video_uid": "f9c337af-fdf9-4737-b4c2-c5f68cd18d6a",
        "ego4d_starting_sec": 210,
        "ego4d_ending_sec": 390,
        "review_file": "../../iclr/replacement_candidates/ego4d_round3/028_de54e5c6-711b-46a2-abeb-be8671870883.mp4",
        "sha256": "ebbcbb41be402ab2b8669c0588fed34dcf8e895c5fbfc2b9124733bd0619a4f9",
    },
    "ego4d-losandes-03": {
        "context_id": "ego4d-6770cb70",
        "q_uid": "6770cb70-0367-4b91-bf7a-211c8c7cb567",
        "google_drive_id": "1KUyuUIYfQPY7wigDxHSaDBC0AEdsXQEY",
        "ego4d_video_uid": "1bfac46e-f957-4495-9583-dbd7fa683225",
        "ego4d_starting_sec": 6420,
        "ego4d_ending_sec": 6600,
        "review_file": "../../iclr/replacement_candidates/ego4d_round3/045_6770cb70-0367-4b91-bf7a-211c8c7cb567.mp4",
        "sha256": "99014e4bb5e35bc6e7f18a2327c87ac497f14ba814f2cafe5f27ee269ed4a92b",
    },
    "ego4d-losandes-04": {
        "context_id": "ego4d-3abe265d",
        "q_uid": "3abe265d-9cad-4e47-856a-722e11997b07",
        "google_drive_id": "1aMArBYEu5QogybFllbq04rHJe118kYU6",
        "ego4d_video_uid": "b7a276c8-6763-45f2-9566-5b3801db70d7",
        "ego4d_starting_sec": 1440,
        "ego4d_ending_sec": 1620,
        "review_file": "../../iclr/replacement_candidates/ego4d_round4/006_3abe265d-9cad-4e47-856a-722e11997b07.mp4",
        "sha256": "e3bd7c14531ed660ce7b347bbb75dd8344d416631b9f2fba6d4c16daf3984fd4",
    },
}


def ego4d_record(record: dict[str, object]) -> dict[str, object]:
    source = str(record["review_file"])
    q_uid = str(record["q_uid"])
    return {
        "context_id": record["context_id"],
        "source_id": f"egoschema:{q_uid}",
        "source_video": source,
        "source_sha256": record["sha256"],
        "source_uri": f"https://drive.google.com/uc?id={record['google_drive_id']}",
        "license": "Ego4D Dataset License Agreement; EgoSchema linked clip",
        "selection_note": (
            "Author-approved outdoor Ego4D replacement selected by visual review; "
            "no model output or metric was inspected during source selection. "
            f"EgoSchema q_uid={q_uid}; Ego4D video_uid="
            f"{record['ego4d_video_uid']}; source range="
            f"{record['ego4d_starting_sec']}--{record['ego4d_ending_sec']} seconds; "
            f"review file={source}."
        ),
        "boundary": {"timestamp_s": 15.0},
        "outdoor_verified": True,
        "outdoor_verification_note": (
            "Approved directly by the author on 2026-09-24 after visual review."
        ),
    }


def main() -> None:
    panel = json.loads(BASE.read_text(encoding="utf-8"))
    spatial_patch = json.loads(SPATIAL_PATCH.read_text(encoding="utf-8"))
    spatial = {row["context_id"]: row for row in spatial_patch["contexts"]}
    contexts = []
    for source in panel["contexts"]:
        old_id = source["context_id"]
        row = copy.deepcopy(source)
        if old_id in EGO4D:
            row.update(ego4d_record(EGO4D[old_id]))
        if old_id in SEKAI:
            row.update(SEKAI[old_id])
        if old_id in spatial:
            replacement = spatial[old_id]
            row.update({
                "source_video": replacement["source_video"],
                "source_sha256": replacement["source_sha256"],
                "source_uri": replacement["source_uri"],
                "boundary": {"timestamp_s": replacement["boundary_timestamp_s"]},
                "selection_note": (
                    "Author-approved clean official SpatialVID clip.mp4; the "
                    "rendered output.mp4 with baked control symbols is invalid."
                ),
                "outdoor_verification_note": (
                    "Start, middle, and end frames were visually checked without "
                    "control symbols and approved by the author on 2026-09-24."
                ),
            })
        contexts.append(row)

    panel.update({
        "panel_id": "outdoor-balanced32-v2",
        "description": (
            "Final balanced outdoor panel: eight contexts each from FrodoBots, "
            "Ego4D, Sekai, and SpatialVID, with author-approved replacements."
        ),
        "selection_protocol": (
            "Sources were selected by visual review before replacement-model "
            "generation. No model outputs or evaluation scores informed the lock."
        ),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "contexts": contexts,
    })
    assert len(contexts) == 32
    assert len({row["context_id"] for row in contexts}) == 32
    assert sum(row["dataset"] == "Ego4D" for row in contexts) == 8
    for row in contexts:
        path = (OUTPUT.parent / row["source_video"]).resolve()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == row["source_sha256"], (row["context_id"], actual)

    payload = json.dumps(panel, indent=2, ensure_ascii=False) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"sha256={hashlib.sha256(payload.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
