#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Generic 10k-campaign full arm. Submit with:
#   sbatch -J <arm>-10k --export=ALL,ARM=<arm> sbatch/train_arm10k.sbatch
# 500 steps @ ~41s/step = ~5.7h; ckpt every 100, rotation off (keep 99).
: "${ARM:?submit with --export=ALL,ARM=<arm>}"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAUNCHER="srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
export TOTAL_STEPS=500
export SAVE_EVERY=100
DRIVER_COPY=$SLURM_SUBMIT_DIR/logs/.driver_${SLURM_JOB_ID}.sh
cp /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh "$DRIVER_COPY"
ARM=$ARM bash "$DRIVER_COPY"
