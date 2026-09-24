#!/usr/bin/env python3
"""Replace normalized Panel32 control decisions from saved motion vectors.

This is a reaggregation utility: it never reads or decodes video.  It applies
the final vector control rule to the saved g0/g1-derived magnitude and cosine,
then updates only control rows in the normalized observation tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from panel32_control_rule import (
    VECTOR_CONTROL_ANGLE_DEGREES,
    VECTOR_CONTROL_CARDINAL_MAGNITUDE_CUTOFF,
    VECTOR_CONTROL_COSINE_CUTOFF,
    VECTOR_CONTROL_DIAGONAL_MAGNITUDE_CUTOFF,
    VECTOR_CONTROL_NOOP_MAGNITUDE_CUTOFF,
    VECTOR_CONTROL_RULE_ID,
    apply_vector_control_rule,
)


WINDOW_STARTS = (0, 6, 12, 18, 24)
KEYS = (
    "context_id", "action", "model_id", "window_start_s", "window_end_s",
    "video_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-control", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    return parser.parse_args()


def prepare_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {
        "uid", "direction", "model", "window_start_s", "window_end_s",
        "video_sha256", "cosine", "magnitude",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"raw control missing columns: {sorted(missing)}")
    raw = raw[raw["window_start_s"].isin(WINDOW_STARTS)].copy()
    raw = apply_vector_control_rule(raw)
    raw = raw.rename(columns={
        "uid": "context_id", "direction": "action", "model": "model_id",
    })
    if raw.duplicated(list(KEYS)).any():
        raise RuntimeError("raw control contains duplicate observation keys")
    expected = raw["model_id"].nunique() * 288 * len(WINDOW_STARTS)
    if len(raw) != expected:
        raise RuntimeError(f"expected {expected} reporting rows, found {len(raw)}")
    return raw


def update_observations(path: Path, decisions: pd.DataFrame) -> dict:
    observations = pd.read_csv(path)
    control_mask = observations["metric"].eq("control")
    control = observations.loc[control_mask].copy()
    lookup = decisions[[*KEYS, "control_failure"]]
    merged = control.merge(lookup, on=list(KEYS), how="left", validate="one_to_one")
    if merged["control_failure"].isna().any():
        missing = merged.loc[merged["control_failure"].isna(), list(KEYS)].head()
        raise RuntimeError(f"{path}: missing raw decisions:\n{missing}")
    observations.loc[control_mask, "failed"] = merged["control_failure"].astype(int).to_numpy()
    # ``semantics`` is the observation-schema class, not the decision-rule
    # identifier; retain ``independent_six_second_window`` and record the
    # exact rule in the dedicated provenance sidecar.
    temporary = path.with_suffix(path.suffix + ".tmp")
    observations.to_csv(temporary, index=False)
    temporary.replace(path)
    return {
        "path": str(path),
        "control_rows": int(control_mask.sum()),
        "control_failures": int(observations.loc[control_mask, "failed"].sum()),
    }


def main() -> None:
    args = parse_args()
    decisions = prepare_raw(args.raw_control)
    outputs = []
    for group in ("main", "ode", "dmd"):
        outputs.append(update_observations(
            args.final_dir / f"panel32_{group}_observations.csv", decisions
        ))

    audit_columns = [
        "context_id", "action", "model_id", "window_start_s", "window_end_s",
        "g0", "g1", "magnitude", "cosine", "angle_failure",
        "weak_magnitude_failure", "noop_magnitude_failure", "control_failure",
        "control_rule_id", "video_sha256",
    ]
    decisions[audit_columns].to_csv(
        args.final_dir / "control_vector_decisions.csv", index=False
    )
    summary = (
        decisions.groupby(["model_id", "window_end_s", "action"], as_index=False)
        .agg(failures=("control_failure", "sum"), total=("control_failure", "size"))
    )
    summary["failure_rate_pct"] = 100.0 * summary["failures"] / summary["total"]
    summary.to_csv(args.final_dir / "control_vector_summary.csv", index=False)
    provenance = {
        "control_rule_id": VECTOR_CONTROL_RULE_ID,
        "angle_degrees": VECTOR_CONTROL_ANGLE_DEGREES,
        "cosine_cutoff": VECTOR_CONTROL_COSINE_CUTOFF,
        "cardinal_magnitude_cutoff": VECTOR_CONTROL_CARDINAL_MAGNITUDE_CUTOFF,
        "diagonal_magnitude_cutoff": VECTOR_CONTROL_DIAGONAL_MAGNITUDE_CUTOFF,
        "noop_magnitude_cutoff": VECTOR_CONTROL_NOOP_MAGNITUDE_CUTOFF,
        "raw_control": str(args.raw_control.resolve()),
        "reporting_rows": len(decisions),
        "updated_observations": outputs,
    }
    (args.final_dir / "control_vector_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
