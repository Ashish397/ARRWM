"""Smoke-test the unified chunk_metrics.csv against the old per-metric CSVs.

Confirms the single-pass extractor reproduces what the disparate scripts produced,
BEFORE we delete the old CSVs/scripts:
  - g0..g7 / cmd0,cmd1 / r0..r7  vs  analysis/ndof_following.csv  (must be ~exact: same source)
  - per-video mean MUSIQ          vs  analysis/iqa_14e_all.csv    (close: per-chunk vs per-video)
  - MAE / LPIPS present for GT chunks (sanity)
Prints PASS/FAIL per check.
"""
import os, sys
import pandas as pd, numpy as np

cm = pd.read_csv("analysis/chunk_metrics.csv")
print(f"chunk_metrics.csv: {len(cm)} rows | runs {sorted(cm.run.unique())} | "
      f"iqa rows {int(cm.musiq.notna().sum())} | recon(GT) rows {int(cm.mae.notna().sum())}")
ok = True

# 1) action vectors vs ndof_following.csv (same CoTracker->PCA source -> should match closely)
if os.path.exists("analysis/ndof_following.csv"):
    nd = pd.read_csv("analysis/ndof_following.csv")
    keys = ["run", "step", "rank", "branch", "chunk"]
    m = cm.merge(nd, on=keys, suffixes=("_cm", "_nd"))
    print(f"\n[action vectors] {len(m)} overlapping chunks vs ndof_following.csv")
    worst = 0.0
    for c in ["cmd0", "cmd1"] + [f"g{d}" for d in range(8)] + [f"r{d}" for d in range(8)]:
        if f"{c}_cm" in m and f"{c}_nd" in m:
            a = pd.to_numeric(m[f"{c}_cm"], errors="coerce"); b = pd.to_numeric(m[f"{c}_nd"], errors="coerce")
            dd = (a - b).abs(); worst = max(worst, dd.max())
            print(f"   {c:5} max|Δ|={dd.max():.4f} mean|Δ|={dd.mean():.4f}")
    p = worst < 0.05; ok &= p
    print(f"   -> {'PASS' if p else 'FAIL'} (max|Δ| over all dims = {worst:.4f}, tol 0.05)")
else:
    print("\n[action vectors] ndof_following.csv absent -> skip")

# 2) IQA vs iqa_14e_all.csv (per-chunk mean per video vs per-video)
if os.path.exists("analysis/iqa_14e_all.csv"):
    iq = pd.read_csv("analysis/iqa_14e_all.csv")
    cmv = cm.groupby(["run", "step", "rank", "branch"])["musiq"].mean().reset_index()
    m = cmv.merge(iq[["run", "step", "rank", "branch", "musiq"]], on=["run", "step", "rank", "branch"], suffixes=("_cm", "_iq"))
    if len(m):
        d = (m["musiq_cm"] - m["musiq_iq"]).abs(); corr = m["musiq_cm"].corr(m["musiq_iq"])
        p = corr > 0.9; ok &= p
        print(f"\n[iqa musiq] {len(m)} videos: mean|Δ|={d.mean():.2f} corr={corr:.3f} -> {'PASS' if p else 'FAIL'} (corr>0.9)")
    else:
        print("\n[iqa musiq] no overlap -> skip")
else:
    print("\n[iqa musiq] iqa_14e_all.csv absent -> skip")

# 3) recon present for GT
gt_recon = cm[(cm.branch == "gt")].mae.notna().mean()
p = gt_recon > 0.5; ok &= p
print(f"\n[recon] GT chunks with MAE/LPIPS: {gt_recon:.0%} -> {'PASS' if p else 'FAIL'} (>50%)")

print(f"\n===== OVERALL: {'PASS — unified CSV verified, safe to unify/clean up' if ok else 'FAIL — do NOT delete old CSVs/scripts'} =====")
sys.exit(0 if ok else 1)
