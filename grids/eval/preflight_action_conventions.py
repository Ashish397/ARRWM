"""Fail-fast control-adapter audit before the full aligned ICLR evaluation.

This intentionally scores only F/B/L/R for a small fixed context subset, but
uses the exact CoTracker/PCA producer and temporal adapter used by the final
control evaluation.  A clear sign inversion is an adapter error and blocks the
full evaluation; a genuinely weak response is reported as inconclusive.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code_release"))

import fleet30s_common as fc  # noqa: E402
from final_v2_control import frames_at, sample_indices  # noqa: E402
import evaluation.ndof_following as teacher  # noqa: E402

teacher.CK = str(ROOT / "code_release" / "preprocessing" / "checkpoints" / "pca_basis.pt")


def state(delta: float, tolerance: float) -> str:
    if delta > tolerance:
        return "correct"
    if delta < -tolerance:
        return "reversed"
    return "inconclusive_low_response"


def shard_model_context_pairs(models: list[str], uids: list[str],
                              shard_index: int,
                              shard_count: int) -> list[tuple[str, str]]:
    """Balance preflight work over the full model/context product.

    Sharding contexts alone creates empty shards whenever the evaluation uses
    more shards than contexts (the panel32 launch uses 48 shards for 32
    contexts).  Empty CSVs then fail to parse during finalisation.  Splitting
    the 15 x 32 product keeps every panel32 shard non-empty and still assigns
    every model/context pair exactly once.
    """
    pairs = [(model, uid) for model in models for uid in uids]
    return pairs[shard_index::shard_count]


def audit_rows(measured: pd.DataFrame, models: list[str], contexts: int,
               tolerance: float, required_names: list[str]) -> pd.DataFrame:
    expected = contexts * len(models) * 4
    if len(measured) != expected:
        raise AssertionError(("preflight row count", len(measured), expected))
    if measured.duplicated(["model", "uid", "direction"]).any():
        raise AssertionError("duplicate preflight rows")
    if set(measured.model) != set(models) or measured.uid.nunique() != contexts:
        raise AssertionError("incomplete preflight model/context product")
    per_model = measured.groupby("model").agg(
        rows=("uid", "size"), contexts=("uid", "nunique"))
    if not (per_model.rows == contexts * 4).all() or not (
            per_model.contexts == contexts).all():
        raise AssertionError(("incomplete per-model preflight coverage",
                              per_model.to_dict("index")))
    per_pair = measured.groupby(["model", "uid"]).direction.agg(
        lambda values: tuple(sorted(values)))
    if not (per_pair == ("B", "F", "L", "R")).all():
        raise AssertionError("incomplete F/B/L/R coverage in preflight")
    report = []
    for model, group in measured.groupby("model"):
        med = group.groupby("direction")[["g0", "g1"]].median()
        throttle_delta = 0.5 * float(med.loc["F", "g0"] - med.loc["B", "g0"])
        yaw_delta = 0.5 * float(med.loc["R", "g1"] - med.loc["L", "g1"])
        report.append(dict(model=model, throttle_delta=throttle_delta,
                           throttle_status=state(throttle_delta, tolerance),
                           yaw_delta=yaw_delta,
                           yaw_status=state(yaw_delta, tolerance)))
    audit = pd.DataFrame(report).sort_values("model")
    reversed_rows = audit[(audit.throttle_status == "reversed") |
                          (audit.yaw_status == "reversed")]
    if len(reversed_rows):
        raise AssertionError(reversed_rows.to_dict("records"))
    required = audit[audit.model.isin(required_names)]
    if len(required) != len(required_names) or not (required.throttle_status == "correct").all() \
            or not (required.yaw_status == "correct").all():
        raise AssertionError(("required command convention not positively verified",
                              required.to_dict("records")))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--uids", default="u31,u37,u48,u00")
    parser.add_argument(
        "--models",
        default="lingbot,dreamx,minwm,matrixgame2,minwm_ode,ours_recovery_base",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--require-positive", default="minwm,minwm_ode")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    uids = args.uids.split(",")
    models = args.models.split(",")
    required_names = [x for x in args.require_positive.split(",") if x]
    if args.finalize_only:
        paths = [args.out / f"action_preflight_rows_shard{i}.csv"
                 for i in range(args.shard_count)]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError([str(path) for path in paths if not path.exists()])
        measured = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        audit = audit_rows(measured, models, len(uids), args.tolerance, required_names)
        measured.sort_values(["model", "uid", "direction"]).to_csv(
            args.out / "action_preflight_rows.csv", index=False)
        audit.to_csv(args.out / "action_preflight_audit.csv", index=False)
        print(audit.to_string(index=False), flush=True)
        (args.out / "ACTION_PREFLIGHT_COMPLETE").touch()
        return

    mean, comp_t, scales = teacher.load_pca()
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda").eval()
    for parameter in cot.parameters():
        parameter.requires_grad_(False)

    rows = []
    selected_pairs = shard_model_context_pairs(
        models, uids, args.shard_index, args.shard_count)
    for model, uid in selected_pairs:
        for direction in ("F", "B", "L", "R"):
            scene = f"{uid}_{direction}"
            path = fc._path(scene, model)
            n, fps = fc.meta(scene, model)
            indices = sample_indices(fc.ctx_of(model), fps, n, 0)
            rgb = frames_at(path, indices)
            video = torch.from_numpy(rgb).to("cuda").float().permute(0, 3, 1, 2).unsqueeze(0)
            with torch.no_grad():
                z = teacher.teacher_read_video(video, cot, mean, comp_t, scales)[1:].cpu().numpy()
            g = np.nanmean(z, axis=0)
            if not np.isfinite(g[:2]).all():
                raise RuntimeError(f"non-finite control readout: {model}/{scene}")
            rows.append(dict(model=model, uid=uid, direction=direction,
                             g0=float(g[0]), g1=float(g[1]), path=path))
            print(model, scene, float(g[0]), float(g[1]), flush=True)

    measured = pd.DataFrame(rows)
    if args.shard_count == 1:
        measured.to_csv(args.out / "action_preflight_rows.csv", index=False)
        audit = audit_rows(measured, models, len(uids), args.tolerance, required_names)
        audit.to_csv(args.out / "action_preflight_audit.csv", index=False)
        print(audit.to_string(index=False), flush=True)
        (args.out / "ACTION_PREFLIGHT_COMPLETE").touch()
    else:
        measured.to_csv(
            args.out / f"action_preflight_rows_shard{args.shard_index}.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
