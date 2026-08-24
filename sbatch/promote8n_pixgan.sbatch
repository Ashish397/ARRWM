#!/bin/bash
#SBATCH --job-name=pixgan8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 8-node REAL pixgan arm (promotion of the 200-step holder run; researcher
# protocol 10:30). Reuses the holder launch path byte-for-byte by acting as
# its own holder: HOLDER=$SLURM_JOB_ID and srun --overlap inside this
# allocation. 500 steps, ckpt/100.
cd /scratch/u6ex/as1748.u6ex/ARRWM
export HOLDER=$SLURM_JOB_ID
export NNODE=8
export PORTOFF=9990
export RUNSTAMP="$(date +%H%M%S)"
export MAXSTEPS=${MAXSTEPS:-500}
export CKPT_EVERY=100
bash sbatch/run_pixgan8n.sh
echo "pixgan8n driver exit=$? $(date)"
