#!/bin/bash
# 10k-generation integrity check — run every ~20 min while gen10k-p* jobs live.
# Verifies: progress (no stall), no errors in shard logs, no stray tmp files
# (dead writers), no duplicate writes (race detector via qa jsonls), no
# corrupt archives (zip test on newest files). Prints ALERT lines on any
# anomaly; prints OK summary otherwise. State in logs/.gen10k_check_state.
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n_10k
ST=logs/.gen10k_check_state
NOW=$(ls $OUTD | grep -c '\.pt$')
PREV=$(cat $ST 2>/dev/null || echo 0); echo $NOW > $ST
RUNNING=$(squeue --me -h -n gen10k-p0,gen10k-p1,gen10k-p2,gen10k-p3,gen10k-p4,gen10k-p5,gen10k-p6,gen10k-p7,gen10k-p8,gen10k-p9 -t RUNNING 2>/dev/null | wc -l)
PENDING=$(squeue --me -h -n gen10k-p0,gen10k-p1,gen10k-p2,gen10k-p3,gen10k-p4,gen10k-p5,gen10k-p6,gen10k-p7,gen10k-p8,gen10k-p9 -t PENDING 2>/dev/null | wc -l)
echo "files=$NOW (+$((NOW-PREV))) jobs: $RUNNING running / $PENDING pending"

# 1. stall: jobs running but no new files since last check
[ "$RUNNING" -gt 0 ] && [ "$NOW" -le "$PREV" ] && \
  echo "ALERT-STALL: $RUNNING jobs running but file count static at $NOW"

# 2. errors in shard logs (this campaign's logs only)
ERR=$(grep -al 'Traceback\|CUDA out of memory\|RuntimeError\|Killed' logs/lmdb14e_dir8n10k_s*.log 2>/dev/null)
[ -n "$ERR" ] && echo "ALERT-ERRORS in: $ERR"

# 3. stray tmp files older than 15 min = dead/hung writer
STRAY=$(find $OUTD -maxdepth 1 -name '*.tmp*' -mmin +15 2>/dev/null | head -5)
[ -n "$STRAY" ] && echo "ALERT-STRAY-TMP: $STRAY"

# 4. duplicate-write detector: any (context,variant) appearing twice across
#    ALL qa jsonls means two processes wrote the same file
DUP=$(cat $OUTD/qa_shard*.jsonl 2>/dev/null | python -c "
import sys, json, collections
c = collections.Counter()
for line in sys.stdin:
    try: r = json.loads(line); c[(r['w'], r['v'])] += 1
    except Exception: pass
d = [k for k, n in c.items() if n > 1]
print(len(d), d[:5] if d else '')")
case "$DUP" in 0*) : ;; *) echo "ALERT-DUPLICATE-WRITES: $DUP" ;; esac

# 5. corruption spot-check: zip-test the 3 newest archives
for f in $(ls -t $OUTD/*.pt 2>/dev/null | head -3); do
  python -c "
import zipfile, sys
try:
    bad = zipfile.ZipFile('$f').testzip()
    sys.exit(1 if bad else 0)
except Exception:
    sys.exit(1)" || echo "ALERT-CORRUPT: $f"
done
echo "check done $(date +%H:%M)"
