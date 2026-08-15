#!/bin/bash
#SBATCH --job-name=hold-2n-6h
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# NODE HOLDER -- the durable substitute for `salloc`.
#
# WHY: `salloc --no-shell` allocations are owned by the submitting process.
# When the agent session is torn down the allocation is CANCELLED (observed:
# jobs 6001818 and 6004082, both "CANCELLED by <uid>" with Start=None after
# ~9.5h queued -- they never received nodes and died with the session).
# A BATCH job is owned by slurmctld, so it survives session restarts.
#
# USE: submit this, wait for it to start, then drive it from anywhere with
#   ALLOC=<this jobid> TAG=t1 STEPS=3 bash sbatch/iact_roll_smoke.sh
# which does `srun --jobid=$ALLOC ...` into the held nodes. Repeat as often
# as needed for the 4h window -- that is the fast-iteration loop the smoke
# turnaround (one queue wait per attempt) otherwise costs.
echo "HOLDER up on $SLURM_JOB_NODELIST at $(date)"
echo "drive with: ALLOC=$SLURM_JOB_ID TAG=t1 STEPS=3 bash sbatch/iact_roll_smoke.sh"
echo "or durably: write a script to logs/.holder_cmd_$SLURM_JOB_ID.sh (runs here)"
# COMMAND-FILE LOOP, not a bare sleep. Driving the holder from the agent's
# shell via a foreground/`setsid` srun proved fragile TWICE: the srun CLIENT
# lives in the agent session, and when that session is torn down (restart) or
# its orphans are reaped (turn end -- observed: stab30's tasks got SIGTERM at
# 00:01:00, exactly when the launching turn ended), slurmstepd kills the step.
# Here the HOLDER polls for a command file and runs it AS the batch job, which
# slurmctld owns -- no agent-side process is load-bearing. Submitting work =
# writing a file; collecting results = reading the log it writes.
CMD=logs/.holder_cmd_$SLURM_JOB_ID.sh
END=$((SECONDS + 21600))
while [ $SECONDS -lt $END ]; do
  if [ -f "$CMD" ]; then
    RUN=logs/.holder_running_$SLURM_JOB_ID.sh
    mv "$CMD" "$RUN"
    echo "HOLDER executing $(date):"; cat "$RUN"
    ALLOC=$SLURM_JOB_ID bash "$RUN"
    echo "HOLDER command exit=$? $(date)"
    mv "$RUN" "logs/.holder_done_$SLURM_JOB_ID.$(date +%s).sh"
  fi
  sleep 10
done
echo "HOLDER done $(date)"
