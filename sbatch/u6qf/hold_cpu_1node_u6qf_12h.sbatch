#!/bin/bash
#SBATCH --job-name=hold-reloc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/%x_%j.err
set -u
L=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s
echo "CPU HOLDER up on $SLURM_JOB_NODELIST at $(date) job=$SLURM_JOB_ID"
CMD=$L/.holder_cmd_$SLURM_JOB_ID.sh
END=$((SECONDS + 42500))
while [ "$SECONDS" -lt "$END" ]; do
  if [ -f "$CMD" ]; then
    RUN=$L/.holder_running_$SLURM_JOB_ID.sh
    mv "$CMD" "$RUN"
    echo "CPU HOLDER executing $(date):"
    cat "$RUN"
    ALLOC=$SLURM_JOB_ID bash "$RUN"
    echo "CPU HOLDER command exit=$? $(date)"
    mv "$RUN" "$L/.holder_done_$SLURM_JOB_ID.$(date +%s).sh"
  fi
  sleep 10
done
echo "CPU HOLDER done $(date)"
