#!/usr/bin/env python3
"""Apply the locked evidence-review decisions to the conjuration-v2 queue.

The rank lists refer to the immutable contact-sheet indices produced from the
merged audit.  They are intentionally converted back to candidate IDs before
writing so a changed candidate set/order fails instead of shifting labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIORITY_POSITIVE_RANKS = (
    1, 4, 7, 10, 11, 12, 14, 21, 23, 25, 28, 34, 35, 36, 40, 47,
    50, 52, 53, 55, 56, 59, 61, 62, 72,
)
MINWM_POSITIVE_RANKS = (1, 4, 9, 11, 19, 25, 30, 59, 67, 70, 249)

# These are the representative bottom-boundary hand/body entries discussed in
# the audit.  They are legitimate entries from outside the image, not interior
# births, and therefore remain negative under the instrument definition.
BOTTOM_ENTRY_IDS = {
    "7fa24d3eeac3e87381e2c162",  # minWM spatialvid-sample18_F
    "d6052be5f79f6d043c3e9735",  # minWM spatialvid-sample20_F
}


def ranked_ids(path: Path, ranks: tuple[int, ...]) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("candidate_ids")
    if not isinstance(values, list) or len(values) != int(payload.get("rows", -1)):
        raise ValueError(f"invalid contact-sheet index: {path}")
    if any(rank < 1 or rank > len(values) for rank in ranks):
        raise ValueError(f"rank outside contact-sheet index: {path}")
    return {str(values[rank - 1]) for rank in ranks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root.resolve()
    target = root / "conjuration_v2_adjudication.csv"
    frame = pd.read_csv(target, keep_default_na=False)
    if frame.candidate_id.duplicated().any():
        raise ValueError("duplicate candidate IDs")
    positives = ranked_ids(
        root / "vlm_review_priority" / "index.json", PRIORITY_POSITIVE_RANKS,
    ) | ranked_ids(
        root / "vlm_review_minwm" / "index.json", MINWM_POSITIVE_RANKS,
    )
    missing = positives - set(frame.candidate_id.astype(str))
    if missing:
        raise ValueError(f"positive decisions absent from candidate queue: {sorted(missing)}")
    frame["adjudication"] = 0
    frame["reason"] = (
        "Rejected after temporal-evidence review: no confirmed abrupt, "
        "persistent interior object birth."
    )
    selected = frame.candidate_id.astype(str).isin(positives)
    frame.loc[selected, "adjudication"] = 1
    frame.loc[selected, "reason"] = (
        "Accepted after temporal-evidence review: absent before birth, then "
        "appears abruptly in the interior and persists."
    )
    bottom = frame.candidate_id.astype(str).isin(BOTTOM_ENTRY_IDS)
    if int(bottom.sum()) != len(BOTTOM_ENTRY_IDS):
        raise ValueError("a locked bottom-entry example is absent")
    frame.loc[bottom, "adjudication"] = 0
    frame.loc[bottom, "reason"] = (
        "Rejected: hand/body enters naturally through the bottom image "
        "boundary rather than appearing in the interior."
    )
    frame.to_csv(target, index=False)
    accepted = frame[frame.adjudication.eq(1)]
    summary = accepted.groupby("model").size().sort_index().to_dict()
    print(json.dumps({
        "candidates": len(frame),
        "accepted_candidates": len(accepted),
        "accepted_videos": int(accepted[["scene", "model"]].drop_duplicates().shape[0]),
        "accepted_by_model": {str(key): int(value) for key, value in summary.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
