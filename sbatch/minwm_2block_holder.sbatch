#!/bin/bash
#SBATCH --job-name=mw2blk-hold
#SBATCH --qos=interactive_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
mkdir -p logs
echo "MINWM two-block HOLDER up on $SLURM_JOB_NODELIST at $(date)"
CMD="logs/.holder_cmd_${SLURM_JOB_ID}.sh"
END=$((SECONDS + 10800))
while [ "$SECONDS" -lt "$END" ]; do
  if [ -f "$CMD" ]; then
    RUN="logs/.holder_running_${SLURM_JOB_ID}.sh"
    mv "$CMD" "$RUN"
    echo "HOLDER executing $(date):"
    cat "$RUN"
    ALLOC="$SLURM_JOB_ID" bash "$RUN"
    rc=$?
    echo "HOLDER command exit=$rc $(date)"
    mv "$RUN" "logs/.holder_done_${SLURM_JOB_ID}.$(date +%s).sh"
  fi
  sleep 5
done
echo "MINWM two-block HOLDER done $(date)"
