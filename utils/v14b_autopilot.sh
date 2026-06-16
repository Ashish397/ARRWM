#!/bin/bash
# Autonomous driver for v14b: wait for the curated pool -> sanity-gate it ->
# auto-launch v14b training -> monitor to completion -> report. Designed to run
# detached in the background; when it exits, the parent agent is re-invoked to
# handle the ODE stage. Idempotent: never double-submits v14b.
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate 2>/dev/null
conda activate arrwm 2>/dev/null

POOL_JOB=5261776
POOL_JSON=paper_assets/v14b_train_windows.json
V14B_SBATCH=sbatch/train_v14b_weunz.sbatch

echo "[autopilot] start $(date)"

# 1) Wait for the curated-pool chain (scoring -> pool) to finish.
while squeue -j "$POOL_JOB" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; do sleep 60; done
echo "[autopilot] pool chain left queue $(date)"
sleep 10  # let the filesystem settle

# 2) Sanity-gate the pool.
if [ ! -f "$POOL_JSON" ]; then
  echo "[autopilot] HOLD: $POOL_JSON missing (pool build failed). Not launching v14b."
  tail -25 logs/build-curated-pool_${POOL_JOB}.out 2>/dev/null
  exit 1
fi
N=$(python -c "import json;print(json.load(open('$POOL_JSON'))['n_windows'])" 2>/dev/null)
NB=$(python -c "import json;d=json.load(open('$POOL_JSON'));print(sum(w['backward'] for w in d['windows']))" 2>/dev/null)
echo "[autopilot] pool n_windows=$N backward_entries=$NB"
if ! [ "$N" -ge 5000 ] 2>/dev/null || ! [ "$N" -le 2000000 ] 2>/dev/null; then
  echo "[autopilot] HOLD: pool size $N outside sane range [5000, 2000000]. Not launching v14b (needs param review)."
  exit 1
fi
echo "[autopilot] pool sane. passes over pool at global64 x2500 = $(python -c "print(round(64*2500/$N,2))")"

# 3) Idempotent submit of v14b.
if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "v14b-weunz"; then
  echo "[autopilot] a v14b-weunz job is already in the queue; not resubmitting."
  JID=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '/v14b-weunz/{print $1; exit}')
else
  JID=$(sbatch --parsable "$V14B_SBATCH")
  echo "[autopilot] submitted v14b training: job $JID $(date)"
fi

# 4) Wait for v14b to finish.
if [ -n "${JID:-}" ]; then
  while squeue -j "$JID" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; do sleep 120; done
  echo "[autopilot] v14b job $JID left queue $(date)"
fi

# 5) Report final state for the agent to pick up (best-ckpt + ODE next).
echo "=== v14b checkpoints ==="
ls -t logs/v14b_weunz/causal_lora_step*.pt 2>/dev/null | head -5
echo "=== v14b log tail ==="
tail -25 logs/v14b-weunz-64gpu_${JID}.out 2>/dev/null
echo "[autopilot] done $(date)"
