#!/usr/bin/env python3
"""Build the author-corrected balanced panel revision from locked v2.

This revision replaces the rejected Ego4D ``0bd9a251`` source with the
author-approved ``a07fc4f3`` clip and moves three explicitly named Ego4D
generation boundaries to 2.0 seconds.  All other sources and boundaries are
copied byte-for-byte from v2.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "grids/eval/panel32_locked_v2.json"
OUTPUT = ROOT / "grids/eval/panel32_locked_v3.json"

REPLACED_CONTEXT = "ego4d-0bd9a251"
REPLACEMENT = {
    "context_id": "ego4d-a07fc4f3",
    "source_id": "egoschema:a07fc4f3-e5bc-4f2a-aa9f-a617f962ebdd",
    "source_video": (
        "../../iclr/replacement_candidates/ego4d_round3/"
        "013_a07fc4f3-e5bc-4f2a-aa9f-a617f962ebdd.mp4"
    ),
    "source_sha256": (
        "120c6445b5a1c1b17d3f5fbfda820c050c5c3c409d01ea44a822a4fc846bc710"
    ),
    "source_uri": "https://drive.google.com/uc?id=1DBbsItzEMmX9AjQBXbRDIM1p2bFuZsEd",
    "license": "Ego4D Dataset License Agreement; EgoSchema linked clip",
    "selection_note": (
        "Author-approved outdoor Ego4D replacement selected by visual review; "
        "no model output or metric was inspected during source selection. "
        "EgoSchema q_uid=a07fc4f3-e5bc-4f2a-aa9f-a617f962ebdd; "
        "Ego4D video_uid=f543a47a-c081-481b-a415-cfe732ad74fe; "
        "source range=1290--1470 seconds; review file=../../iclr/"
        "replacement_candidates/ego4d_round3/"
        "013_a07fc4f3-e5bc-4f2a-aa9f-a617f962ebdd.mp4."
    ),
    "boundary": {"timestamp_s": 15.0},
    "outdoor_verified": True,
    "outdoor_verification_note": (
        "Approved directly by the author on 2026-09-24 after visual review."
    ),
}

BOUNDARY_OVERRIDES = {
    "ego4d-8ed9e028": 2.0,
    "ego4d-de54e5c6": 2.0,
    "ego4d-3abe265d": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    panel = json.loads(BASE.read_text(encoding="utf-8"))
    contexts = []
    for original in panel["contexts"]:
        row = copy.deepcopy(original)
        if row["context_id"] == REPLACED_CONTEXT:
            row.update(REPLACEMENT)
        if row["context_id"] in BOUNDARY_OVERRIDES:
            timestamp = BOUNDARY_OVERRIDES[row["context_id"]]
            row["boundary"] = {"timestamp_s": timestamp}
            row["selection_note"] = (
                str(row.get("selection_note", "")).rstrip()
                + f" Generation boundary author-corrected to {timestamp:.1f} seconds."
            )
        contexts.append(row)

    panel.update({
        "panel_id": "outdoor-balanced32-v3",
        "description": (
            "Author-corrected final balanced outdoor panel: eight contexts "
            "each from FrodoBots, Ego4D, Sekai, and SpatialVID."
        ),
        "created_utc": "2026-09-24T00:00:00Z",
        "contexts": contexts,
    })

    assert len(contexts) == 32
    assert len({row["context_id"] for row in contexts}) == 32
    assert sum(row["dataset"] == "Ego4D" for row in contexts) == 8
    assert REPLACED_CONTEXT not in {row["context_id"] for row in contexts}
    assert REPLACEMENT["context_id"] in {row["context_id"] for row in contexts}
    for row in contexts:
        source = (OUTPUT.parent / row["source_video"]).resolve()
        actual = sha256_file(source)
        assert actual == row["source_sha256"], (row["context_id"], actual)

    payload = json.dumps(panel, indent=2, ensure_ascii=False) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"sha256={hashlib.sha256(payload.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
