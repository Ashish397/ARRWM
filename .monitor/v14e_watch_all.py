#!/usr/bin/env python
"""Watch ALL v14e runs and exit (re-invoking the agent) at the first actionable
event across any of them.

  per-job phase=start: exit on  terminal | clean start (>=5 steps) | setup hang
  per-job phase=run:   exit on  terminal | NaN/inf loss | step stall

Args: jobid:name:run[:phase]  [...]   (phase defaults to start)
"""
import sys, time, subprocess, glob, math

# each arg: jobid:name:run[:phase]
JOBS = []
for a in sys.argv[1:]:
    p = a.split(":")
    if len(p) == 3:
        p = p + ["start"]
    JOBS.append(p)                                    # [jobid,name,run,phase]
LOGDIR = "/scratch/u6ex/as1748.u6ex/ARRWM/logs"
POLL, MAX_ITERS = 120, 250
HANG_S, STALL_S = 50 * 60, 25 * 60


def state(job):
    try:
        out = subprocess.run(["sacct", "-j", job, "-n", "-o", "State,Elapsed", "-X"],
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


def tail(job, name, n=40):
    out = []
    for f in sorted(glob.glob(f"{LOGDIR}/{name}_{job}.err") + glob.glob(f"{LOGDIR}/{name}_{job}.out")):
        out.append(f"--- {f} ---")
        try:
            out.append("".join(open(f, errors="ignore").readlines()[-n:]))
        except Exception as e:
            out.append(f"(read err {e})")
    return "\n".join(out)


def wandb_probe(run):
    try:
        import wandb
        api = wandb.Api(timeout=40)
        rs = list(api.runs("ashish397-university-of-exeter/frodobots_wm", filters={"display_name": run}))
        if not rs:
            return -1, None
        r = sorted(rs, key=lambda x: x.created_at)[-1]
        rows = list(r.scan_history())
        if not rows:
            return -1, None
        mx = max((d.get("_step", -1) for d in rows), default=-1)
        for d in rows[-8:]:
            for k, v in d.items():
                if not k.startswith("_") and isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)):
                    return mx, f"{k}={v}@step{d.get('_step')}"
        return mx, None
    except Exception:
        return -1, None


def main():
    laststep = {j[0]: -1 for j in JOBS}
    lastchg = {j[0]: time.time() for j in JOBS}
    for _ in range(MAX_ITERS):
        for job, name, run, phase in JOBS:
            st, el = state(job)
            if any(t in st for t in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF", "COMPLETED", "DEADLINE")):
                print(f"TERMINAL {name} ({job}): state={st} elapsed={el}s phase={phase}\n")
                print(tail(job, name))
                return 0
            if "RUNNING" in st and phase == "resume":
                # log-based confirmation (avoids the dual-wandb-run step confusion):
                # the trainer prints "Resuming from ...stepNNN.pt" + "start_step=NNN"
                # only AFTER ~20min of setup. Fire once that appears, or on a bad resume.
                import re
                txt = ""
                for f in (glob.glob(f"{LOGDIR}/{name}_{job}.err") + glob.glob(f"{LOGDIR}/{name}_{job}.out")):
                    try:
                        txt += open(f, errors="ignore").read()
                    except Exception:
                        pass
                if "Skipping resume" in txt or "nothing to train" in txt:
                    print(f"RESUME PROBLEM {name} ({job}): did NOT load a checkpoint\n{tail(job,name)}")
                    return 0
                ss = re.search(r"start_step=(\d+)", txt)
                if ss and int(ss.group(1)) > 0:
                    rf = re.search(r"Resuming from .*?(step0*\d+\.pt)", txt)
                    print(f"RESUME CONFIRMED {name} ({job}): start_step={ss.group(1)} "
                          f"({rf.group(1) if rf else '?'})")
                    return 0
                continue
            if "RUNNING" in st:
                step, nan = wandb_probe(run)
                if nan:
                    print(f"BREAK {name} ({job}): non-finite {nan} step={step}\n")
                    print(tail(job, name))
                    return 0
                if step > laststep[job]:
                    laststep[job], lastchg[job] = step, time.time()
                if phase == "start":
                    if step >= 5:
                        print(f"CLEAN START {name} ({job}): step={step} elapsed={el}s")
                        return 0
                    if el > HANG_S and step < 1:
                        print(f"SUSPECT HANG {name} ({job}): RUNNING {el}s no steps\n")
                        print(tail(job, name))
                        return 0
                elif step >= 1 and (time.time() - lastchg[job]) > STALL_S:
                    print(f"SUSPECT STALL {name} ({job}): step stuck {step} >{STALL_S//60}min\n")
                    print(tail(job, name))
                    return 0
        time.sleep(POLL)
    print(f"watch window elapsed; steps={laststep} -- re-arm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
