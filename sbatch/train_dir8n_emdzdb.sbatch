#!/bin/bash
#SBATCH --job-name=dir8n-emdzdb
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Zero-init transport head trained against a VICReg/GECO-style SATISFIABLE
# deadband (lit-review 2026-08-07): per-channel log-ratio vs the PREVIOUS
# committed chunk, squared hinge below 0 (contraction) and above +0.05
# (runaway), zero gradient inside -> target is ZERO decay (beats the
# teacher's k~0.986). Plus identity-minimality so correction stays minimal.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAUNCHER="srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
export TOTAL_STEPS=500
ARM=emdzdb bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh
