import sys, time, subprocess
JOBS=dict(j.split("=") for j in sys.argv[1:])
def st(j):
    try: return subprocess.run(["sacct","-j",j,"-n","-o","State","-X"],capture_output=True,text=True,timeout=30).stdout.strip().split("\n")[0].strip()
    except: return "?"
def term(s): return any(t in s for t in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","NODE_FAIL","OUT_OF","DEADLINE"))
for _ in range(180):
    states={lab:st(j) for j,lab in JOBS.items()}
    if all(term(s) for s in states.values()):
        print("ALL DONE:", states); break
    time.sleep(120)
else:
    print("timeout; states:", {l:st(j) for j,l in JOBS.items()})
