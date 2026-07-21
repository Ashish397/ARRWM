#!/usr/bin/env python
"""Continuously watch the v14e run and exit (re-invoking the agent) at the
first actionable event. Two phases:

  start (default): exit on terminal state | clean start (>=5 steps) | setup hang
  run:             exit on terminal state | NaN/inf loss | step stall | job gone

The agent re-arms after each exit (start -> run -> ... ) so coverage is
continuous from queue through the whole 4h run. On timeout it just re-arms.

Args: JOB_ID [phase]
"""
import sys, time, subprocess, glob, math

JOB = sys.argv[1]
PHASE = sys.argv[2] if len(sys.argv) > 2 else "start"
NAME = "v14e-pca8"
LOGDIR = "/scratch/u6ex/as1748.u6ex/ARRWM/logs"
RUN = "v14e-pca8-raw"
POLL, MAX_ITERS = 120, 250          # ~8.3h window, then re-arm
RUN_NO_STEP_HANG_S = 50 * 60
STALL_S = 25 * 60                    # running but step unchanged this long => suspect


def state():
    try:
        out = subprocess.run(["sacct", "-j", JOB, "-n", "-o", "State,Elapsed", "-X"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        if not out:
            return "?", 0
        p = out.split("\n")[0].split()
        st = p[0] if p else "?"
        el = p[1] if len(p) > 1 else "0:0:0"
        h, m, s = (el.split("-")[-1].split(":") + ["0", "0", "0"])[:3]
        return st, int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return "?", 0


def tail_logs(n=45):
    out = []
    for f in sorted(glob.glob(f"{LOGDIR}/{NAME}_{JOB}.err") + glob.glob(f"{LOGDIR}/{NAME}_{JOB}.out")):
        out.append(f"--- tail {f} ---")
        try:
            out.append("".join(open(f, errors="ignore").readlines()[-n:]))
        except Exception as e:
            out.append(f"(read err {e})")
    return "\n".join(out)


def wandb_probe():
    """Return (max_step, nan_key_or_None)."""
    try:
        import wandb
        api = wandb.Api(timeout=40)
        rs = list(api.runs("ashish397-university-of-exeter/frodobots_wm",
                           filters={"display_name": RUN}))
        if not rs:
            return -1, None
        r = sorted(rs, key=lambda x: x.created_at)[-1]
        rows = list(r.scan_history())
        if not rows:
            return -1, None
        mx = max((d.get("_step", -1) for d in rows), default=-1)
        for d in rows[-8:]:                                   # recent NaN/inf scan
            for k, v in d.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                    return mx, f"{k}={v}@step{d.get('_step')}"
        return mx, None
    except Exception:
        return -1, None


def main():
    last_step, last_change = -1, time.time()
    for _ in range(MAX_ITERS):
        st, elapsed = state()
        if any(t in st for t in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF", "COMPLETED", "DEADLINE")):
            print(f"TERMINAL: state={st} elapsed={elapsed}s phase={PHASE}\n")
            print(tail_logs())
            return 0
        if "RUNNING" in st:
            step, nan = wandb_probe()
            if nan:
                print(f"BREAK: non-finite loss {nan} (state={st}, step={step})\n")
                print(tail_logs())
                return 0
            if step > last_step:
                last_step, last_change = step, time.time()
            if PHASE == "start":
                if step >= 5:
                    print(f"CLEAN START: step={step} elapsed={elapsed}s")
                    return 0
                if elapsed > RUN_NO_STEP_HANG_S and step < 1:
                    print(f"SUSPECT HANG: RUNNING {elapsed}s, no steps\n")
                    print(tail_logs())
                    return 0
            else:  # run
                if step >= 1 and (time.time() - last_change) > STALL_S:
                    print(f"SUSPECT STALL: step stuck at {step} for >{STALL_S//60}min\n")
                    print(tail_logs())
                    return 0
        time.sleep(POLL)
    print(f"watch window elapsed (phase={PHASE}, last_step={last_step}) -- re-arm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
