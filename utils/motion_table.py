"""Motion-magnitude comparison table across runs, from motion_check.csv.

Replaces ad-hoc parsing of [mc] blocks out of slurm logs, which silently
mis-attributed rows between runs (2026-08-07: three different serve
configs printed byte-identical magnitudes — a parser artifact, not a
real result). motion_check.py already writes a per-(run, dir, seed) CSV;
this reads THAT and reports magnitudes vs a baseline run, so the numbers
are traceable to their source rows.

Run motion_check.py FIRST with every run in one MC_RUNS invocation, so a
single CSV holds them all:
    MC_RUNS=a:b:c python utils/motion_check.py
    MT_BASE=a python utils/motion_table.py

Env: MT_CSV (default flow_viz/motion_check.csv), MT_BASE (baseline run),
MT_RUNS (optional colon list / order filter).
"""
import os
import csv
from collections import defaultdict

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
CSV = os.environ.get("MT_CSV", f"{FV}/motion_check.csv")
BASE = os.environ.get("MT_BASE", "")
ONLY = [r for r in os.environ.get("MT_RUNS", "").split(":") if r]
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]


def main():
    per = defaultdict(dict)                    # run -> dir -> [(fwd, ste)]
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            per[row["run"]].setdefault(row["dir"], []).append(
                (float(row["fwd"]), float(row["steer"])))
    runs = ONLY or sorted(per)
    missing = [r for r in runs if r not in per]
    if missing:
        print(f"[mt] NOT IN CSV (rerun motion_check with them): {missing}")
    runs = [r for r in runs if r in per]
    if not runs:
        raise SystemExit(f"[mt] no runs in {CSV}")

    def mags(run):
        d = per[run]
        have = [x for x in DIRS if x in d]
        f = [abs(sum(v for v, _ in d[x]) / len(d[x])) for x in have]
        s = [abs(sum(v for _, v in d[x]) / len(d[x])) for x in have]
        return sum(f) / len(f), sum(s) / len(s), len(have)

    base = BASE if BASE in per else runs[0]
    bf, bs, _ = mags(base)
    print(f"[mt] {CSV}\n[mt] baseline = {base}\n")
    print(f"{'run':28} {'|fwd|':>7} {'|steer|':>8} {'dirs':>5}   vs baseline")
    for r in runs:
        f, s, n = mags(r)
        tag = "  (baseline)" if r == base else (
            f"   fwd {100*(f/bf-1):+6.1f}%  steer {100*(s/bs-1):+6.1f}%")
        warn = "  <-- INCOMPLETE" if n < len(DIRS) else ""
        print(f"{r:28} {f:7.3f} {s:8.3f} {n:5d}{tag}{warn}")
    # Guard against the exact failure that motivated this tool.
    seen = {}
    for r in runs:
        key = tuple(round(x, 6) for x in mags(r)[:2])
        if key in seen:
            print(f"\n[mt] WARNING: {r} and {seen[key]} have IDENTICAL "
                  f"magnitudes {key} — check for duplicated recordings.")
        seen[key] = r


if __name__ == "__main__":
    main()
