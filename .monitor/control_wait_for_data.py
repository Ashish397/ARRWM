#!/usr/bin/env python
"""Block until a control-test run has logged >= MIN_STEPS, then exit so the
agent is re-invoked to run control_lr_check.py.  Args: MIN_STEPS BASELINE_JSON.

BASELINE is a comma list of "name:already_verified_max_step"; we only trip when
some run advances PAST its baseline (so re-arming after a partial check waits
for genuinely new data instead of firing immediately).
"""
import sys, time, json
import wandb

ENT, PROJ = "ashish397-university-of-exeter", "frodobots_wm"
NAMES = ["v14d-control-test-4node", "v14d-control-test-8node", "v14d-control-test-16node"]
MIN_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BASELINE = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
MAX_ITERS, SLEEP = 160, 180  # ~8h cap


def maxstep(api, name):
    runs = list(api.runs(f"{ENT}/{PROJ}", filters={"display_name": name}))
    if not runs:
        return -1
    r = sorted(runs, key=lambda x: x.created_at)[-1]
    s = -1
    for d in r.scan_history(keys=["_step"]):
        if d.get("_step") is not None:
            s = max(s, d["_step"])
    return s


def main():
    for _ in range(MAX_ITERS):
        try:
            api = wandb.Api(timeout=60)
            hits = []
            for n in NAMES:
                ms = maxstep(api, n)
                base = BASELINE.get(n, MIN_STEPS - 1)
                if ms >= MIN_STEPS and ms > base:
                    hits.append((n, ms))
            if hits:
                print("DATA READY:", hits)
                return 0
        except Exception as e:
            print("poll error:", e)
        time.sleep(SLEEP)
    print("waiter timed out")
    return 1


if __name__ == "__main__":
    sys.exit(main())
