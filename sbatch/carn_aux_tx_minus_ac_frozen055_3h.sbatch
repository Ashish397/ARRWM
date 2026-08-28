#!/bin/bash
#SBATCH --job-name=carn-aux-txm-ac-f055
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

PRINT_ONLY=0 \
HOLDER="$SLURM_JOB_ID" \
CARNVX_MODE=aux_tx_minus_ac_frozen055 \
PORTOFF=28632 \
MAXSTEPS=300 \
RUNSTAMP=actionablate2808 \
bash sbatch/run_carn_vx_on_holder.sh
