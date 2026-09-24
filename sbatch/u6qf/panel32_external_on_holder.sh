#!/bin/bash
# Inject one model/action into an existing four-GPU holder allocation.
set -euo pipefail
MODEL=${1:?usage: ALLOC=job panel32_external_on_holder.sh MODEL ACTION MANIFEST PROVENANCE}
ACTION=${2:?usage: ALLOC=job panel32_external_on_holder.sh MODEL ACTION MANIFEST PROVENANCE}
MANIFEST=${3:?usage: ALLOC=job panel32_external_on_holder.sh MODEL ACTION MANIFEST PROVENANCE}
PROVENANCE=${4:?usage: ALLOC=job panel32_external_on_holder.sh MODEL ACTION MANIFEST PROVENANCE}
ALLOC=${ALLOC:-${HOLDER:-}}
test -n "$ALLOC"
STATE=$(squeue -h -j "$ALLOC" -o %T | head -n1)
test "$STATE" = RUNNING
NODE=${PANEL32_NODE:-$(squeue -h -j "$ALLOC" -o %N | head -n1)}
R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
LOG=${PANEL32_EXTERNAL_LOG:-${PANEL32_STAGE:-$R/panel32_stage}/logs/$MODEL/$ACTION}
mkdir -p "$LOG"
srun --jobid="$ALLOC" --overlap --external-launcher --nodes=1 --nodelist="$NODE" \
  --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --output="$LOG/srun.out" --error="$LOG/srun.err" \
  bash "$A/sbatch/u6qf/panel32_external_holder_action.sh" \
    "$MODEL" "$ACTION" "$MANIFEST" "$PROVENANCE"
