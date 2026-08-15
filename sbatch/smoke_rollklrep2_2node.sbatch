#!/bin/bash
#SBATCH --job-name=smoke-rollklrep2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# SMOKE for the KV-cache AR rollout ODE stage. 2 nodes = 8 ranks, which is the
# MINIMUM: _CurriculumBatches bin-packs to exactly `world`, so with world < the
# max group size (8) it returns an empty batch list and cycle() spins forever.
# Purpose: (1) does it run at all after the 6 review fixes, (2) MEASURE the real
# step time -- 33 small sequential forwards per branch instead of one packed
# forward, so the wall-clock cost is the open question, not memory.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAUNCHER="srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
export TOTAL_STEPS=30
export SAVE_EVERY=30
DRIVER_COPY=$SLURM_SUBMIT_DIR/logs/.driver_${SLURM_JOB_ID}.sh
cp /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh "$DRIVER_COPY"
ARM=rollklrep2smoke bash "$DRIVER_COPY"
