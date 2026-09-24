#!/usr/bin/env python3
"""Fail-closed merge of independent conjuration-v2 VLM triage shards."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranked", type=Path, required=True)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates, keep_default_na=False)
    if candidates.candidate_id.astype(str).duplicated().any():
        raise ValueError("candidate IDs are not unique")
    expected = candidates.candidate_id.astype(str).tolist()
    parts = []
    for shard in range(args.shard_count):
        path = args.shards / f"shard{shard}.csv"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, keep_default_na=False)
        expected_shard = [candidate for index, candidate in enumerate(expected)
                          if index % args.shard_count == shard]
        if frame.candidate_id.astype(str).tolist() != expected_shard:
            raise ValueError(f"shard {shard} does not contain its exact ordered candidate set")
        if not frame.p_conjuration_vlm.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"shard {shard} contains invalid probabilities")
        if not frame.vlm_flag.astype(int).isin([0, 1]).all():
            raise ValueError(f"shard {shard} contains invalid flags")
        for row in frame.itertuples(index=False):
            evidence = Path(str(row.evidence_path))
            if not evidence.is_file() or sha256(evidence) != str(row.evidence_sha256):
                raise ValueError(f"evidence changed after triage: {row.candidate_id}")
        parts.append(frame)
    scores = pd.concat(parts, ignore_index=True)
    if scores.candidate_id.astype(str).duplicated().any() or set(scores.candidate_id.astype(str)) != set(expected):
        raise ValueError("merged triage is not a one-to-one cover of candidates")
    lookup = scores.set_index(scores.candidate_id.astype(str), verify_integrity=True)
    ordered = lookup.loc[expected].reset_index(drop=True)
    output = candidates.merge(
        ordered[["candidate_id", "p_conjuration_vlm", "vlm_flag",
                 "model_id", "prompt_sha256", "evidence_sha256"]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    if output.p_conjuration_vlm.isna().any():
        raise ValueError("missing triage score after merge")
    ranked = output.sort_values(
        ["p_conjuration_vlm", "legacy_positive", "peak", "candidate_id"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    ranked.to_csv(args.ranked, index=False)
    print({
        "candidates": len(output),
        "vlm_positive": int(output.vlm_flag.astype(int).sum()),
        "p_ge_0.25": int((output.p_conjuration_vlm >= 0.25).sum()),
        "p_ge_0.10": int((output.p_conjuration_vlm >= 0.10).sum()),
        "output": str(args.output), "ranked": str(args.ranked),
    })


if __name__ == "__main__":
    main()
