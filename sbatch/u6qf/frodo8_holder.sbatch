#!/bin/bash
# Four-GPU, high-memory command holder for one Frodo8 model.  Submit one
# holder per model so all five model fleets can run concurrently; inject
# `bash sbatch/u6qf/frodo8_on_holder.sh MODEL` through the printed command
# file.  The generation launcher uses one four-GPU srun task, whose node-local
# parent assigns one explicit CUDA ordinal to each of four workers.
#SBATCH --job-name=frodo8-hold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=460000M
#SBATCH --time=02:00:00
#SBATCH --output=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/frodo8-holder-%j.out
#SBATCH --error=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler/aligned32_stage/logs/frodo8-holder-%j.err
set -u

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
LOG="$R/aligned32_stage/logs"
mkdir -p "$LOG"
CMD="$LOG/.frodo8_holder_cmd_${SLURM_JOB_ID}.sh"
END=$((SECONDS + 7000))
echo "FRODO8_HOLDER_READY job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST command=$CMD $(date -Is)"
while [ "$SECONDS" -lt "$END" ]; do
  if [ -f "$CMD" ]; then
    RUN="$LOG/.frodo8_holder_running_${SLURM_JOB_ID}.sh"
    mv "$CMD" "$RUN"
    echo "FRODO8_HOLDER_EXECUTING $RUN $(date -Is)"
    ALLOC="$SLURM_JOB_ID" bash "$RUN"
    rc=$?
    DONE="$LOG/.frodo8_holder_done_${SLURM_JOB_ID}_$(date +%s)_rc${rc}.sh"
    mv "$RUN" "$DONE"
    echo "FRODO8_HOLDER_COMMAND_DONE rc=$rc file=$DONE $(date -Is)"
  fi
  sleep 5
done
echo "FRODO8_HOLDER_EXPIRED job=$SLURM_JOB_ID $(date -Is)"
