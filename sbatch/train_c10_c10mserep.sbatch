#!/bin/bash
#SBATCH --job-name=c10-c10mserep
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# HARD-DIRECTION CURRICULUM, 10 epochs — MSE + BOUNDED gaussian exclusion.
# Epoch 0 trains all 8 compass directions per (context, chunk) with the
# no-op chain as the clean branch (so all 9 get gradient). From epoch 1
# each group contributes only its WORST directions by recorded per-sample
# error plus an equal number of well-learned ones (opposite compass where
# possible) for anti-forgetting: 2 bad -> 4, 1 bad -> 2, 0 bad -> 2 random.
# Measured: epoch 0 = 618 steps, later epochs ~309 -> ~3400 steps ~ 19.6h
# worst case at 20.8 s/step, hence the 20h wall AND periodic checkpoints
# (save_interval=700) so a wall-clock kill still leaves a usable model.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAUNCHER="srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
export TOTAL_STEPS=6000            # ceiling only; the curriculum stops at 10 epochs
export SAVE_EVERY=700
DRIVER_COPY=$SLURM_SUBMIT_DIR/logs/.driver_${SLURM_JOB_ID}.sh
cp /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh "$DRIVER_COPY"
ARM=c10mserep bash "$DRIVER_COPY"
