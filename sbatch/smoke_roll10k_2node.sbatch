#!/bin/bash
#SBATCH --job-name=smoke-roll10k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 2-node smoke of the 10k-scene rollout arm (3 steps) — gates the 4 full
# arms via --dependency=afterok. First checks the generation actually
# delivered: target 10,003 contexts x 9 = 90,027 files minus ~3% span-cap
# skips; require >= 85,000.
OUTD=/projects/u6ex/fbots/frodobots_lmdb/v14e_pilot_dir8n_10k
NF=$(ls $OUTD | grep -c '\.pt$')
if [ "$NF" -lt 85000 ]; then
  echo "SMOKE-GATE: only $NF/90027 files in $OUTD — generation incomplete"; exit 1
fi
echo "SMOKE-GATE: $NF files present, proceeding"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAUNCHER="srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
export TOTAL_STEPS=3
export SAVE_EVERY=100
DRIVER_COPY=$SLURM_SUBMIT_DIR/logs/.driver_${SLURM_JOB_ID}.sh
cp /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh "$DRIVER_COPY"
ARM=roll10ksmoke bash "$DRIVER_COPY"
