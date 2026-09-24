#!/bin/bash
# Generate this holder's disjoint subset, wait for peer holders, then evaluate.
set -euo pipefail
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
WORKER="$R/minwm_seed29_worker.sh"
EVAL="$R/minwm_seed29_eval.sh"
OUT="$R/minwm_seed29/ARRWM/logs/eval_final/fleet30s/minwm_seed29"

MW_WORKER_ID=node0 MW_ASSIGNED_UIDS=u31,u37,u48,u00,u04,u01,a00,a19 \
  "$WORKER"

deadline=$((SECONDS + 10800))
while [ "$SECONDS" -lt "$deadline" ]; do
  videos=$(find "$OUT" -maxdepth 1 -name '*.mp4' | wc -l)
  sidecars=$(find "$OUT" -maxdepth 1 -name '*.mp4.json' | wc -l)
  echo "WAIT_FOR_PEERS videos=$videos sidecars=$sidecars $(date -Is)"
  if [ "$videos" -eq 288 ] && [ "$sidecars" -eq 288 ]; then
    exec "$EVAL"
  fi
  sleep 20
done
echo "timed out waiting for peer holders" >&2
exit 1
