#!/usr/bin/env python3
"""Build the provisional v2 source manifest from approved source replacements.

This manifest is intentionally marked ``working`` until the replacement Ego4D
contexts are approved.  It is safe for generating the already-approved clean
SpatialVID and Sekai contexts; it must not be used for final aggregation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "grids/eval/panel32_locked_v1.json"
SPATIAL_PATCH = ROOT / "grids/eval/spatialvid_clean_source_patch_v2.json"
OUTPUT = ROOT / "grids/eval/panel32_working_spatialvid_sekai_v2.json"


SEKAI = {
    "sekai-walk-01": {
        "context_id": "sekai-cheongju",
        "source_id": "kqkUJZllt0U_0005550_0007350",
        "source_video": "../../analysis/panel32_v1/sources/sekai_approved/cheongju.mp4",
        "source_sha256": "cf46819202e8188ba20a9568da0795602b3761d1c9ab6e54d022b982c13d5af4",
        "source_uri": "https://www.youtube.com/watch?v=kqkUJZllt0U#t=185.000000",
        "selection_note": (
            "Author-approved replacement from official Sekai Walking-HQ row "
            "kqkUJZllt0U_0005550_0007350; outdoor-urban; Cheongju, Korea; "
            "only the 15-second review range was downloaded."
        ),
        "boundary": {"timestamp_s": 7.5},
        "outdoor_verified": True,
        "outdoor_verification_note": (
            "Visible forward-facing outdoor street throughout the reviewed range; "
            "approved by the author on 2026-09-24."
        ),
    },
    "sekai-walk-07": {
        "context_id": "sekai-krakow",
        "source_id": "SpBSfZMoZYQ_0084087_0085887",
        "source_video": "../../analysis/panel32_v1/sources/sekai_approved/krakow.mp4",
        "source_sha256": "f306d5c38354aba94e3ae88982adb261392a536570be6b26d5c93f26ec09de60",
        "source_uri": "https://www.youtube.com/watch?v=SpBSfZMoZYQ#t=2802.900000",
        "selection_note": (
            "Author-approved replacement from official Sekai Walking-HQ row "
            "SpBSfZMoZYQ_0084087_0085887; outdoor-urban; Krakow, Poland; "
            "only the 15-second review range was downloaded."
        ),
        "boundary": {"timestamp_s": 7.5},
        "outdoor_verified": True,
        "outdoor_verification_note": (
            "Visible forward-facing outdoor street throughout the reviewed range; "
            "approved by the author on 2026-09-24."
        ),
    },
    "sekai-walk-08": {
        "context_id": "sekai-yerevan",
        "source_id": "cOKHJEvZmeE_0046763_0048563",
        "source_video": "../../analysis/panel32_v1/sources/sekai_approved/yerevan.mp4",
        "source_sha256": "774a714d218f449f63981cea904a028a39410db31062b126359095c3a8d67b4f",
        "source_uri": "https://www.youtube.com/watch?v=cOKHJEvZmeE#t=1558.766667",
        "selection_note": (
            "Author-approved replacement from official Sekai Walking-HQ row "
            "cOKHJEvZmeE_0046763_0048563; outdoor-urban; Yerevan, Armenia; "
            "only the 15-second review range was downloaded."
        ),
        "boundary": {"timestamp_s": 7.5},
        "outdoor_verified": True,
        "outdoor_verification_note": (
            "Visible forward-facing outdoor street throughout the reviewed range; "
            "approved by the author on 2026-09-24."
        ),
    },
}


def main() -> None:
    panel = json.loads(BASE.read_text(encoding="utf-8"))
    patch = json.loads(SPATIAL_PATCH.read_text(encoding="utf-8"))
    spatial = {row["context_id"]: row for row in patch["contexts"]}
    contexts = []
    for source in panel["contexts"]:
        row = copy.deepcopy(source)
        if row["context_id"] in SEKAI:
            row.update(SEKAI[row["context_id"]])
        if row["context_id"] in spatial:
            replacement = spatial[row["context_id"]]
            row.update({
                "source_video": replacement["source_video"],
                "source_sha256": replacement["source_sha256"],
                "source_uri": replacement["source_uri"],
                "boundary": {"timestamp_s": replacement["boundary_timestamp_s"]},
                "selection_note": (
                    "Author-approved clean official SpatialVID clip.mp4 replacement; "
                    "the rendered output.mp4 with baked control symbols is invalid."
                ),
                "outdoor_verification_note": (
                    "Start, middle, and end frames visually checked without control "
                    "symbols; approved by the author on 2026-09-24."
                ),
            })
        contexts.append(row)
    panel.update({
        "panel_id": "outdoor-balanced32-working-spatialvid-sekai-v2",
        "description": (
            "Working panel with approved clean SpatialVID sources and approved "
            "Cheongju, Krakow, and Yerevan Sekai replacements. Ego4D rows remain "
            "provisional and this manifest is not eligible for final aggregation."
        ),
        "selection_protocol": (
            "Working source contract for approved SpatialVID and Sekai reruns only. "
            "The final v2 lock will be issued after author approval of replacement "
            "Ego4D contexts; no model results influenced source selection."
        ),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "contexts": contexts,
    })
    payload = json.dumps(panel, indent=2, ensure_ascii=False) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"sha256={hashlib.sha256(payload.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
