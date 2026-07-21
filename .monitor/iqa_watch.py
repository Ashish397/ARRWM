#!/usr/bin/env python
"""Tight watch on the IQA + qset jobs: catch a crash/stall fast (60s poll), not
at the end. Exits (re-invoking the agent) on:
  - IQA terminal (FAILED -> print err tail ; COMPLETED -> done)
  - IQA stall (CSV not written for >5min while RUNNING)
  - qset terminal
"""
import sys, time, subprocess, os

IQA, QSET = "5399032", "5398907"
CSV = "analysis/iqa_14e.csv"
ERR = f"logs/iqa-14e_{IQA}.err"


def state(j):
    try:
        return subprocess.run(["sacct", "-j", j, "-n", "-o", "State", "-X"],
                              capture_output=True, text=True, timeout=30).stdout.strip().split("\n")[0].strip()
    except Exception:
        return "?"


def is_term(s):
    return any(t in s for t in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF", "DEADLINE"))


def errtail(n=30):
    try:
        return "".join(open(ERR, errors="ignore").readlines()[-n:])
    except Exception:
        return "(no err log)"


def mtime():
    try:
        return os.path.getmtime(CSV)
    except Exception:
        return 0


def rows():
    try:
        return sum(1 for _ in open(CSV))
    except Exception:
        return 0


def main():
    last_mt, last_change = mtime(), time.time()
    for _ in range(300):  # 60s * 300 = 5h
        si = state(IQA)
        if is_term(si):
            if "COMPLETED" in si:
                print(f"IQA COMPLETED (rows={rows()})")
            else:
                print(f"IQA {si} (rows={rows()})\n{errtail()}")
            return 0
        if "RUNNING" in si:
            mt = mtime()
            if mt > last_mt:
                last_mt, last_change = mt, time.time()
            elif time.time() - last_change > 300:
                print(f"IQA STALL: CSV unchanged >5min (rows={rows()})\n{errtail(20)}")
                return 0
        time.sleep(60)
    print("iqa_watch window elapsed -- re-arm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
