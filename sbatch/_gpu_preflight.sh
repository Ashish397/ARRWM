#!/bin/bash
# GPU PRE-FLIGHT for a holder allocation.
#
# WHY: gan2x2_raw_t0 (6102572) and gan2x2_wave_ts (6102573) both died ~80 s
# after launch with `torch.OutOfMemoryError` at `_initialize_kv_cache`, on
# rank 4 = the SECOND node's local rank 0, with GPU 0 reporting ~1-2 GiB free
# of 95 GiB while our own process held only 17-37 GiB. Something else owned
# the rest of that GPU. Each failure then wasted the remaining ~5.9 h of a
# 6 h holder, because the holder's command file had already been consumed.
#
# WHAT: report per-node/per-GPU used memory, reap OUR OWN stale python
# processes, re-check, and FAIL FAST (non-zero) if any GPU is still dirty --
# so the caller can retry or move on instead of training into an OOM.
#
# Usage: HOLDER=<jobid> [NNODE=2] [DIRTY_GB=5] bash sbatch/_gpu_preflight.sh
set -uo pipefail
: "${HOLDER:?set HOLDER to the holder job id}"
NNODE=${NNODE:-2}
DIRTY_GB=${DIRTY_GB:-5}
NODES=${NODES:-$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | head -"$NNODE" | paste -sd,)}

echo "[preflight] holder=$HOLDER nodes=$NODES dirty_threshold=${DIRTY_GB}GB $(date)"

_scan() {
  srun --jobid="$HOLDER" --overlap --nodelist="$NODES" --nodes="$NNODE" \
       --ntasks-per-node=1 --gpu-bind=none bash -c '
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "[preflight] $(hostname) gpu_used_MiB: $(echo "$used" | tr "\n" " ")"
    echo "[preflight] $(hostname) my_python: $(pgrep -u "$USER" -c python 2>/dev/null || echo 0)"
  ' 2>&1 | grep "^\[preflight\]"
}

echo "--- before ---"; _scan

# Reap only OUR OWN leftovers. A fresh allocation should have none; anything
# here is a zombie from a previous step on the same node.
srun --jobid="$HOLDER" --overlap --nodelist="$NODES" --nodes="$NNODE" \
     --ntasks-per-node=1 --gpu-bind=none bash -c \
     'pkill -u "$USER" -f "trainer/causal_action_forcing_train.py" 2>/dev/null; \
      pkill -u "$USER" python 2>/dev/null; sleep 5; exit 0' >/dev/null 2>&1

echo "--- after reap ---"
OUT=$(_scan); echo "$OUT"

# Any GPU above the threshold => dirty node.
BAD=$(echo "$OUT" | awk -v lim="$((DIRTY_GB*1024))" '
  /gpu_used_MiB/ { for (i=4; i<=NF; i++) if ($i+0 > lim) { print "DIRTY " $2 " " $i; } }')
if [ -n "$BAD" ]; then
  echo "[preflight] FAIL -- GPU memory still held after reap:"
  echo "$BAD"
  exit 3
fi
echo "[preflight] OK -- all GPUs on $NODES are clean"
