#!/bin/bash
#SBATCH --job-name=hold-ours30s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=03:30:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/%x_%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s/%x_%j.err
# NODE HOLDER for the 30 s DMD-ablation fleets (u6qf). Same command-file loop as
# sbatch/hold_nodes_2node.sbatch: a batch job owned by slurmctld, driven by writing
# _logs/ours30s/.holder_cmd_<jobid>.sh, which it runs as ALLOC=<jobid>.
L=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/_logs/ours30s
echo "HOLDER up on $SLURM_JOB_NODELIST at $(date) job=$SLURM_JOB_ID"
CMD=$L/.holder_cmd_$SLURM_JOB_ID.sh
END=$((SECONDS + 12000))
while [ $SECONDS -lt $END ]; do
  if [ -f "$CMD" ]; then
    RUN=$L/.holder_running_$SLURM_JOB_ID.sh; mv "$CMD" "$RUN"
    echo "HOLDER executing $(date):"; cat "$RUN"
    ALLOC=$SLURM_JOB_ID bash "$RUN"; echo "HOLDER command exit=$? $(date)"
    mv "$RUN" "$L/.holder_done_$SLURM_JOB_ID.$(date +%s).sh"
  fi
  sleep 10
done
echo "HOLDER done $(date)"
