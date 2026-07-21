import sys, time, subprocess
JOBS = dict(j.split("=") for j in sys.argv[1:])  # id=label
def term(j):
    try:
        o=subprocess.run(["sacct","-j",j,"-n","-o","State","-X"],capture_output=True,text=True,timeout=30).stdout.strip().split("\n")[0].strip()
        return o, any(t in o for t in ("COMPLETED","FAILED","TIMEOUT","CANCELLED","NODE_FAIL","OUT_OF","DEADLINE"))
    except: return "?",False
done=set()
for _ in range(180):  # ~6h
    for j,lab in JOBS.items():
        if j in done: continue
        st,t=term(j)
        if t:
            print(f"DONE {lab} ({j}): {st}"); done.add(j)
    if len(done)==len(JOBS): break
    if done: break  # report first completion, re-arm for rest
    time.sleep(120)
