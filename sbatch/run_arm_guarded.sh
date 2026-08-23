#!/bin/bash
# Guarded wrapper around sbatch/run_full_carn_probe.sh.
#
# Two failure modes cost a full 6 h holder each on 2026-08-23:
#   1. a dirty GPU on the holder's SECOND node -> OOM at KV-cache init ~80 s in
#      (gan2x2_raw_t0 / gan2x2_wave_ts);
#   2. the holder's command file is consumed on first read, so a fast failure
#      leaves the allocation idle for the rest of its walltime.
# This wrapper pre-flights the GPUs, retries once after re-reaping, and logs a
# per-rank device/free-memory line so any repeat is diagnosable from the log.
set -uo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${HOLDER:?}" ; : "${DARM:?}"
ATTEMPTS=${ATTEMPTS:-2}

for try in $(seq 1 "$ATTEMPTS"); do
  echo "[guard] === $DARM attempt $try/$ATTEMPTS $(date) ==="
  if ! HOLDER="$HOLDER" bash sbatch/_gpu_preflight.sh; then
    echo "[guard] preflight FAILED on attempt $try"
    if [ "$try" -lt "$ATTEMPTS" ]; then sleep 60; continue; fi
    echo "[guard] giving up: holder nodes are not usable for $DARM"
    exit 3
  fi
  # RUNSTAMP must differ per attempt so a retry never writes into the
  # half-populated run dir of the attempt that just failed.
  export RUNSTAMP="$(date +%H%M%S)"
  if bash sbatch/run_full_carn_probe.sh; then
    echo "[guard] $DARM SUCCEEDED on attempt $try $(date)"
    exit 0
  fi
  echo "[guard] $DARM FAILED on attempt $try (see logs/exp_${DARM}.err)"
  tail -5 "logs/exp_${DARM}.err" 2>/dev/null
done
echo "[guard] $DARM exhausted $ATTEMPTS attempts $(date)"
exit 1
