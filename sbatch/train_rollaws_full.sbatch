#!/bin/bash
#SBATCH --job-name=rollaws
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# FULL KV-cache AR rollout run: the student trained the way 14e is INFERENCED.
# 8 dirs + no-op data, hard-direction curriculum, 10 epochs.
# Dataset after the duplicate fix: 9,893 chains / 1,237 groups -> epoch 0 is
# ~309 steps at a 32 global batch, later epochs ~154. 10 epochs ~ 1,700 steps.
# Step time is UNKNOWN until the smoke reports, so: 24h wall + frequent
# checkpoints (every 200 steps) so a wall-clock kill still leaves a usable model
# and we can probe partial epochs.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAUNCHER="srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
export TOTAL_STEPS=4000        # ceiling only; the curriculum stops at 10 epochs
export SAVE_EVERY=200
export CURRIC_EPOCHS=10
DRIVER_COPY=$SLURM_SUBMIT_DIR/logs/.driver_${SLURM_JOB_ID}.sh
cp /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh "$DRIVER_COPY"
ARM=rollaws bash "$DRIVER_COPY"
