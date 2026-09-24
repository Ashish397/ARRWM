#!/bin/bash
# Reusable four-GPU panel32 command holder.  Each array element owns one node;
# injected commands may run sequentially until the allocation expires.
#SBATCH --job-name=panel32-hold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000M
#SBATCH --time=04:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/panel32_stage/logs/holders/%A_%a.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/panel32_stage/logs/holders/%A_%a.err
set -u
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
L="$R/panel32_stage/logs/holders"
mkdir -p "$L"
CMD="$L/.holder_cmd_${SLURM_JOB_ID}.sh"
END=$((SECONDS + 14100))
echo "PANEL32_HOLDER_READY job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST command=$CMD $(date -Is)"
while [ "$SECONDS" -lt "$END" ]; do
  if [ -f "$CMD" ]; then
    RUN="$L/.holder_running_${SLURM_JOB_ID}.sh"
    mv "$CMD" "$RUN"
    echo "PANEL32_HOLDER_EXECUTING $RUN $(date -Is)"
    ALLOC="$SLURM_JOB_ID" bash "$RUN"
    rc=$?
    DONE="$L/.holder_done_${SLURM_JOB_ID}_$(date +%s)_rc${rc}.sh"
    mv "$RUN" "$DONE"
    echo "PANEL32_HOLDER_COMMAND_DONE rc=$rc file=$DONE $(date -Is)"
  fi
  sleep 5
done
echo "PANEL32_HOLDER_EXPIRED job=$SLURM_JOB_ID $(date -Is)"
