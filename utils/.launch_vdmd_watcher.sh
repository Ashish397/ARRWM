#!/bin/bash
# Submit the DMD viz watcher ONLY once a dmd10k arm is actually RUNNING.
# Review finding A5: submitted early, the watcher (2 nodes) starts long before
# the six 8-node --exclusive arms and burns its whole window polling for
# checkpoints that do not exist yet, then exits having rendered nothing.
cd /scratch/u6ex/as1748.u6ex/ARRWM
for i in $(seq 1 720); do            # up to ~12h of waiting
  if squeue --me -h -t RUNNING -o '%j' | grep -q '^dmd10k-'; then
    JID=$(sbatch --parsable sbatch/viz_watcherdmd.sbatch)
    echo "[vdmd-launch] arm running -> submitted watcher $JID at $(date)"
    exit 0
  fi
  sleep 60
done
echo "[vdmd-launch] gave up waiting $(date)"
