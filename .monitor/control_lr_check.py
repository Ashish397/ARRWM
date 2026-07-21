#!/usr/bin/env python
"""Verify the control-test runs' REALIZED hyperparameters match 14d's, scaled
by the per-run sqrt(batch) factor.

14d (bukeu8pa) logs `train/learning_rate` and `train/critic_learning_rate`,
both linearly warmed 0->peak over 200 steps then ~flat (cosine/30000).  Each
control run must track 14d's curve multiplied by its batch factor:

    4-node  (batch 16): factor 0.5
    8-node  (batch 32): factor 0.7071
    16-node (batch 64): factor 1.0   (== 14d exactly)

A realized deviation beyond TOL from that scaled curve is a genuine input-
hyperparameter mismatch -> the run should be killed and the config fixed.

Usage: python .monitor/control_lr_check.py [TOL]   (default TOL=0.03)
"""
import sys
import numpy as np
import wandb

ENT = "ashish397-university-of-exeter"
PROJ = "frodobots_wm"
REF_ID = "bukeu8pa"  # v14d
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03

RUNS = [
    ("v14d-control-test-4node", 0.5,    16),
    ("v14d-control-test-8node", 0.70710678, 32),
    ("v14d-control-test-16node", 1.0,   64),
]
KEYS = ["train/learning_rate", "train/critic_learning_rate", "grad_norm/lora"]


def series(run, key):
    pts = []
    for d in run.scan_history(keys=["_step", key]):
        if d.get("_step") is not None and d.get(key) is not None:
            pts.append((d["_step"], d[key]))
    pts.sort()
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def main():
    api = wandb.Api(timeout=60)
    ref = api.run(f"{ENT}/{PROJ}/{REF_ID}")
    ref_lr = {k: series(ref, k) for k in KEYS[:2]}

    overall_ok = True
    for name, factor, batch in RUNS:
        runs = [r for r in api.runs(f"{ENT}/{PROJ}", filters={"display_name": name})]
        runs = [r for r in runs if r.state != "crashed" or True]
        print(f"\n===== {name}  (batch {batch}, expected factor x{factor:.4f}) =====")
        if not runs:
            print("  not started yet (no wandb run).")
            continue
        run = sorted(runs, key=lambda r: r.created_at)[-1]
        print(f"  run id {run.id}  state={run.state}")
        for key in KEYS[:2]:
            cs, cv = series(run, key)
            if len(cs) == 0:
                print(f"  {key}: no data yet")
                continue
            rs, rv = ref_lr[key]
            # 14d value interpolated at the control run's steps
            exp = np.interp(cs, rs, rv) * factor
            good = exp > 0
            rel = np.abs(cv[good] - exp[good]) / np.maximum(exp[good], 1e-12)
            mx = rel.max() if rel.size else 0.0
            last = cs[-1]
            status = "OK" if mx <= TOL else "*** MISMATCH ***"
            if mx > TOL:
                overall_ok = False
            print(f"  {key:28s} steps<= {int(last):4d}  realized[-1]={cv[-1]:.3e}  "
                  f"expected[-1]={exp[-1]:.3e}  max_dev={mx*100:5.2f}%  {status}")
        gs, gv = series(run, "grad_norm/lora")
        if len(gs):
            print(f"  grad_norm/lora: last={gv[-1]:.4f} median={np.median(gv):.4f} "
                  f"(14d median ~0.02-0.03; flag if >>1 or NaN)")

    print("\n" + ("ALL RUNS MATCH 14d (within %.0f%%)" % (TOL * 100) if overall_ok
                  else "AT LEAST ONE MISMATCH -> investigate / kill+fix"))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
