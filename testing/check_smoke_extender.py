"""Post-job analysis for the extender smoke (sbatch/smoke_extender.sbatch).

Pulls the two phase runs from wandb (smoke_extender_low_j<JOB> and
smoke_extender_high_j<JOB>), compares ``gen/mae_extension_count``
between phases, and asserts:

  1) LOW phase has substantially HIGHER avg extension count than HIGH
     phase (extender extends on easy data, cuts off on hard data).
  2) Sample videos were uploaded to wandb (sample/* keys present).
  3) Local mp4 sample files exist on disk for both phases.

Exit code: 0 on PASS, 1 on FAIL. Designed to be run interactively
after the sbatch finishes:

    python tests/check_smoke_extender.py --job_id <SLURM_JOB_ID>

Or pass --auto to find the latest pair of smoke runs in the project.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb

PROJECT = "ashish397-university-of-exeter/longlive-phase1-smoke"
LOGROOT = Path("/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_extender")
# Streaming extender key (new code path, lives in
# trainer._streaming_maybe_extend). The legacy non-streaming path logs
# under gen/mae_extension_count instead — switch the key if you change
# the smoke config back to streaming_mode=false.
EXT_KEY = "gen/streaming_extension_count"
BASE_KEY = "gen/baseline_last_chunk_mae"
LAST_KEY = "gen/streaming_extension_last_mae"
SAMPLE_KEY = "sample/pred_image"

# Heuristic thresholds. Low-motion rides should average several
# extensions per iter (typical: ~6-13 with cap=13). High-motion should
# average 0-2 (extender cuts off in 1-2 chunks once per-chunk MAE
# crosses 0.5). The factor is loose enough that noise from a 6-iter
# run won't trip it.
MIN_LOW_AVG = 3.0
MAX_HIGH_AVG = 3.0
MIN_RATIO = 1.5


def find_runs(api: wandb.Api, job_id: str | None) -> tuple[wandb.apis.public.Run, wandb.apis.public.Run]:
    runs = list(api.runs(PROJECT, filters={"display_name": {"$regex": "^smoke_extender_"}}))
    if not runs:
        raise SystemExit(f"No smoke runs found under {PROJECT}.")
    if job_id is not None:
        wanted = {f"smoke_extender_low_j{job_id}", f"smoke_extender_high_j{job_id}"}
        matched = [r for r in runs if r.name in wanted]
        if len(matched) != 2:
            raise SystemExit(
                f"Expected both runs {sorted(wanted)} but found {[r.name for r in matched]}.",
            )
        low = next(r for r in matched if "low" in r.name)
        high = next(r for r in matched if "high" in r.name)
        return low, high
    # Auto: pair the most-recent low and high.
    low_runs = sorted(
        [r for r in runs if "low" in r.name], key=lambda r: r.created_at, reverse=True,
    )
    high_runs = sorted(
        [r for r in runs if "high" in r.name], key=lambda r: r.created_at, reverse=True,
    )
    if not low_runs or not high_runs:
        raise SystemExit(
            f"Need at least one low + one high run; found low={len(low_runs)} high={len(high_runs)}",
        )
    return low_runs[0], high_runs[0]


def avg_metric(run: wandb.apis.public.Run, key: str) -> tuple[float, int]:
    hist = run.history(keys=[key, "_step"], samples=2000, pandas=True)
    if hist is None or key not in hist or hist[key].dropna().empty:
        return float("nan"), 0
    vals = hist[key].dropna().astype(float)
    return float(vals.mean()), int(len(vals))


def has_video(run: wandb.apis.public.Run) -> bool:
    return SAMPLE_KEY in run.summary or any(
        SAMPLE_KEY in str(k) for k in run.summary.keys()
    )


def local_mp4_count(phase: str) -> int:
    p = LOGROOT / f"phase_{phase}" / "samples"
    if not p.exists():
        return 0
    return len(list(p.glob("step_*.mp4")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", type=str, default=None,
                    help="SLURM job id; if omitted, picks the latest pair of smoke runs.")
    args = ap.parse_args()

    api = wandb.Api()
    low_run, high_run = find_runs(api, args.job_id)
    print(f"LOW  run: {low_run.name} ({low_run.id}) state={low_run.state}")
    print(f"HIGH run: {high_run.name} ({high_run.id}) state={high_run.state}")

    fails: list[str] = []

    # ----- 1. Extension count -----
    low_avg, low_n = avg_metric(low_run, EXT_KEY)
    high_avg, high_n = avg_metric(high_run, EXT_KEY)
    print(f"\n[ext] {EXT_KEY}: LOW avg={low_avg:.2f} (n={low_n})  HIGH avg={high_avg:.2f} (n={high_n})")

    if low_n == 0 or high_n == 0:
        fails.append(
            f"FAIL: missing {EXT_KEY} in one or both runs (low_n={low_n}, high_n={high_n}). "
            "Check that streaming_mode=true + mae_extension_threshold/max_extra_chunks "
            "are set in the smoke config so _streaming_maybe_extend actually runs."
        )
    else:
        if low_avg < MIN_LOW_AVG:
            fails.append(
                f"FAIL: LOW extension avg {low_avg:.2f} < expected min {MIN_LOW_AVG}. "
                "Extender did not extend on easy data."
            )
        if high_avg > MAX_HIGH_AVG:
            fails.append(
                f"FAIL: HIGH extension avg {high_avg:.2f} > expected max {MAX_HIGH_AVG}. "
                "Extender did not cut off on hard data."
            )
        if low_avg < MIN_RATIO * max(high_avg, 0.1):
            fails.append(
                f"FAIL: LOW/HIGH ratio {low_avg / max(high_avg, 0.1):.2f} < {MIN_RATIO}. "
                "The two phases didn't separate enough."
            )

    # ----- 2. Sanity: baseline / last chunk MAE both logged -----
    base_low, n_b_low = avg_metric(low_run, BASE_KEY)
    base_high, n_b_high = avg_metric(high_run, BASE_KEY)
    print(f"[mae] {BASE_KEY}: LOW={base_low:.3f} (n={n_b_low})  HIGH={base_high:.3f} (n={n_b_high})")
    if n_b_low == 0 or n_b_high == 0:
        fails.append(
            f"FAIL: {BASE_KEY} missing in one or both runs (low_n={n_b_low}, high_n={n_b_high})."
        )

    # ----- 3. Wandb video upload -----
    low_has_vid = has_video(low_run)
    high_has_vid = has_video(high_run)
    print(f"\n[video-wandb] LOW has sample/*: {low_has_vid}  HIGH has sample/*: {high_has_vid}")
    if not low_has_vid:
        fails.append("FAIL: LOW run missing sample/* in wandb (vis_to_wandb path).")
    if not high_has_vid:
        fails.append("FAIL: HIGH run missing sample/* in wandb (vis_to_wandb path).")

    # ----- 4. Local mp4 files -----
    n_low = local_mp4_count("low")
    n_high = local_mp4_count("high")
    print(f"[video-local] mp4s on disk: LOW={n_low}  HIGH={n_high}")
    # With sample_interval=2 and sample_at_steps=[1] over max_steps=6, expect 4 mp4s per phase
    # (steps 1, 2, 4, 6).
    if n_low == 0:
        fails.append("FAIL: no LOW mp4s on disk (vis_save_local path).")
    if n_high == 0:
        fails.append("FAIL: no HIGH mp4s on disk (vis_save_local path).")

    print()
    if fails:
        print("=" * 60)
        print("SMOKE FAILED")
        print("=" * 60)
        for f in fails:
            print(f"  - {f}")
        return 1
    print("=" * 60)
    print(
        f"SMOKE PASSED — extender extends on low-motion ({low_avg:.1f} avg) "
        f"and cuts off on high-motion ({high_avg:.1f} avg); "
        f"videos uploaded to wandb + saved locally ({n_low} low, {n_high} high mp4s)."
    )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
