"""Validate dual-reference pilot controls before fleet-wide scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["scene", "model", "window_start_s"]


def paired(d: pd.DataFrame, condition: str, column: str) -> pd.DataFrame:
    base = d[d.condition == "baseline"][KEY + [column]].rename(columns={column: "baseline"})
    ctrl = d[d.condition == condition][KEY + [column]].rename(columns={column: "control"})
    z = base.merge(ctrl, on=KEY, validate="one_to_one")
    z["delta"] = z.control - z.baseline
    return z


def result(name: str, z: pd.DataFrame, minimum_median: float,
           minimum_positive_fraction: float, absolute: bool = False) -> dict:
    delta = z.delta.abs() if absolute else z.delta
    median = float(delta.median())
    fraction = float((delta > 0).mean())
    return {"check": name, "n": len(z), "median_delta": median,
            "positive_fraction": fraction,
            "minimum_median_delta": minimum_median,
            "minimum_positive_fraction": minimum_positive_fraction,
            "absolute_delta": absolute,
            "passed": bool(median >= minimum_median and
                           fraction >= minimum_positive_fraction)}


def validate_style(d: pd.DataFrame) -> list[dict]:
    checks = [
        result("swapped seed changes seed drift",
               paired(d, "swapped_seed", "seed_dino_drift"), 0.03, 0.625,
               absolute=True),
        result("swapped prior changes rolling drift",
               paired(d, "swapped_prior", "rolling_dino_drift"), 0.03, 0.625),
    ]
    # Reference isolation is exact: changing only one reference must not alter
    # the score attached to the other reference.
    a = paired(d, "swapped_seed", "rolling_dino_drift")
    b = paired(d, "swapped_prior", "seed_dino_drift")
    checks += [
        {"check": "swapped seed leaves rolling drift unchanged", "n": len(a),
         "max_abs_delta": float(a.delta.abs().max()), "maximum_abs_delta": 1e-5,
         "passed": bool(a.delta.abs().max() <= 1e-5)},
        {"check": "swapped prior leaves seed drift unchanged", "n": len(b),
         "max_abs_delta": float(b.delta.abs().max()), "maximum_abs_delta": 1e-5,
         "passed": bool(b.delta.abs().max() <= 1e-5)},
    ]
    return checks


def validate_geometry(d: pd.DataFrame) -> list[dict]:
    warp = paired(d, "warped_target", "p_geometry_absolute")
    # A saturated baseline cannot rise further, so sensitivity is assessed on
    # pilot examples for which the clean target is not already called corrupt.
    warp = warp[warp.baseline < 0.5]
    checks = [
        result("swapped prior raises rolling discontinuity",
               paired(d, "swapped_prior", "p_geometry_rolling_break"), 0.10, 0.625),
        result("warped target raises absolute corruption",
               warp, 0.10, 0.625),
    ]
    a = paired(d, "swapped_prior", "p_geometry_absolute")
    checks.append({
        "check": "swapped prior leaves target-only score unchanged", "n": len(a),
        "max_abs_delta": float(a.delta.abs().max()), "maximum_abs_delta": 1e-5,
        "passed": bool(a.delta.abs().max() <= 1e-5),
    })
    base = d[d.condition == "baseline"]
    for col in ["p_geometry_absolute"]:
        interior = float(((base[col] > 1e-4) & (base[col] < 1 - 1e-4)).mean())
        checks.append({"check": f"{col} is not uniformly saturated", "n": len(base),
                       "interior_fraction": interior, "minimum_interior_fraction": 0.25,
                       "passed": bool(interior >= 0.25)})
    swapped = d[d.condition == "swapped_prior"]
    baseline_clean = float((base.p_geometry_rolling_break < 0.5).mean())
    swapped_flagged = float((swapped.p_geometry_rolling_break > 0.5).mean())
    checks += [
        {"check": "rolling continuity accepts contiguous pilot windows", "n": len(base),
         "clean_fraction": baseline_clean, "minimum_clean_fraction": 0.75,
         "passed": bool(baseline_clean >= 0.75)},
        {"check": "rolling continuity rejects unrelated consecutive references",
         "n": len(swapped), "flagged_fraction": swapped_flagged,
         "minimum_flagged_fraction": 0.75,
         "passed": bool(swapped_flagged >= 0.75)},
    ]
    if "p_geometry_seed_conditioned" in base:
        effect = base.p_geometry_seed_conditioned - base.p_geometry_absolute
        checks.append({"check": "seed-conditioning effect audit", "n": len(base),
                       "median_signed_delta": float(effect.median()),
                       "median_absolute_delta": float(effect.abs().median()),
                       "passed": True})
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    style, geometry = pd.read_csv(a.style), pd.read_csv(a.geometry)
    checks = validate_style(style) + validate_geometry(geometry)
    report = {"status": "PASS" if all(x["passed"] for x in checks) else "FAIL",
              "checks": checks}
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
