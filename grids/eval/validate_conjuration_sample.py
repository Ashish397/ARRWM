#!/usr/bin/env python3
"""Validate the user-sampled minWM conjuration positives and hard negatives.

The detector audit is a high-recall proposal stage, so hard negatives may be
proposed for review.  Before adjudication, this checker requires proposal
recall on all labelled positives.  With an adjudication CSV, it additionally
requires exact agreement on all twelve video-level labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    required = {
        "filename", "scene", "model", "expected_conjuration",
        "validation_group", "rationale",
    }
    require(required.issubset(sample.columns),
            f"sample missing columns: {sorted(required - set(sample.columns))}")
    require(len(sample) == 12, f"expected 12 sampled videos, found {len(sample)}")
    require(not sample.duplicated(["scene", "model"]).any(),
            "sample contains duplicate scene/model rows")
    labels = pd.to_numeric(sample.expected_conjuration, errors="coerce")
    require(labels.notna().all() and labels.isin([0, 1]).all(),
            "expected_conjuration must be binary")
    sample["expected_conjuration"] = labels.astype(int)
    require(int(labels.sum()) == 6, "sample must contain exactly six positives")
    require(int((labels == 0).sum()) == 6, "sample must contain exactly six negatives")

    if args.videos is not None:
        missing_videos = [
            name for name in sample.filename
            if not (args.videos / str(name)).is_file()
        ]
        require(not missing_videos, f"missing sampled videos: {missing_videos}")

    rows = []
    missing_records = []
    for item in sample.itertuples(index=False):
        record_path = args.records / f"{item.scene}__{item.model}.json"
        if not record_path.is_file():
            missing_records.append(str(record_path))
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        require(record.get("audit") == "panel32_conjuration_v2",
                f"wrong audit schema: {record_path}")
        require(record.get("scene") == item.scene and record.get("model") == item.model,
                f"record identity mismatch: {record_path}")
        candidates = [
            candidate
            for window in record.get("windows", [])
            for candidate in window.get("candidates", [])
        ]
        rows.append({
            "filename": item.filename,
            "scene": item.scene,
            "model": item.model,
            "expected_conjuration": int(item.expected_conjuration),
            "validation_group": item.validation_group,
            "candidate_count": len(candidates),
            "candidate_classes": sorted({str(row["cls"]) for row in candidates}),
            "candidate_birth_s": sorted(float(row["birth_s"]) for row in candidates),
            "proposal_recovered": int(bool(candidates)),
        })

    require(not missing_records,
            f"audit is incomplete for {len(missing_records)} sampled videos")
    result = pd.DataFrame(rows)
    missed = result[
        (result.expected_conjuration == 1) & (result.proposal_recovered == 0)
    ]
    require(missed.empty,
            "positive proposal-recall failures: " + ", ".join(missed.scene))

    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "proposal_recall_pass",
        "selection": "user_curated_nonrandom_sanity_check",
        "sampled_videos": len(result),
        "positive_videos": 6,
        "hard_negative_videos": 6,
        "positive_proposals_recovered": int(result.loc[
            result.expected_conjuration == 1, "proposal_recovered"
        ].sum()),
        "hard_negative_proposals_retained_for_review": int(result.loc[
            result.expected_conjuration == 0, "proposal_recovered"
        ].sum()),
        "rows": result.to_dict("records"),
    }

    if args.adjudication is not None:
        adjudication = pd.read_csv(args.adjudication, keep_default_na=False)
        needed = {"scene", "model", "adjudication"}
        require(needed.issubset(adjudication.columns),
                f"adjudication missing columns: {sorted(needed - set(adjudication.columns))}")
        verdict = pd.to_numeric(adjudication.adjudication, errors="coerce")
        require(verdict.notna().all() and verdict.isin([0, 1]).all(),
                "every adjudication must be binary")
        adjudication = adjudication.assign(adjudication=verdict.astype(int))
        predicted = (
            adjudication.groupby(["scene", "model"], as_index=False)
            .adjudication.max()
            .rename(columns={"adjudication": "predicted_conjuration"})
        )
        checked = sample.merge(predicted, on=["scene", "model"], how="left")
        checked["predicted_conjuration"] = (
            checked.predicted_conjuration.fillna(0).astype(int)
        )
        errors = checked[
            checked.expected_conjuration != checked.predicted_conjuration
        ]
        require(errors.empty,
                "adjudication disagrees with sampled labels: " + ", ".join(errors.scene))
        true_positive = int(((checked.expected_conjuration == 1) &
                             (checked.predicted_conjuration == 1)).sum())
        true_negative = int(((checked.expected_conjuration == 0) &
                             (checked.predicted_conjuration == 0)).sum())
        summary.update({
            "status": "adjudication_pass",
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": 0,
            "false_negative": 0,
            "positive_recall": true_positive / 6,
            "hard_negative_specificity": true_negative / 6,
            "balanced_sample_accuracy": (true_positive + true_negative) / 12,
        })

    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
